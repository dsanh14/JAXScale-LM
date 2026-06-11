"""Causal multi-head self-attention with three execution modes.

1. **Training / full sequence** (``cache=None``): queries, keys and values
   all come from the input block; a lower-triangular mask enforces
   causality.
2. **Prefill** (``cache`` given, ``length`` 0): identical math to training,
   but rotated keys/values are also written into the cache at positions
   ``0..S-1``.
3. **Incremental decode** (``cache`` given, ``S == 1``): one new token's
   K/V is written at position ``length``; the query attends over the full
   fixed-capacity cache with positions ``>= length+1`` masked out.

All three share one code path: the only differences are the source of the
key/value tensors and the mask. Softmax runs in float32 regardless of the
compute dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.model.cache import LayerCache, update_layer
from jaxscale_lm.model.embeddings import apply_rope, rope_angles
from jaxscale_lm.utils.tree import dtype_from_name

_MASK_VALUE = -1e30  # large finite negative; -inf can produce NaNs via inf-inf


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention with RoPE and optional KV cache."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        param_dtype = dtype_from_name(config.parameter_dtype)
        compute_dtype = dtype_from_name(config.compute_dtype)
        init = nnx.initializers.normal(config.initializer_range)

        def linear(out_features: int) -> nnx.Linear:
            return nnx.Linear(
                config.hidden_size,
                out_features,
                use_bias=config.use_bias,
                kernel_init=init,
                param_dtype=param_dtype,
                dtype=compute_dtype,
                rngs=rngs,
            )

        self.q_proj = linear(self.num_heads * self.head_dim)
        self.k_proj = linear(self.num_kv_heads * self.head_dim)
        self.v_proj = linear(self.num_kv_heads * self.head_dim)
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            use_bias=config.use_bias,
            kernel_init=init,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            rngs=rngs,
        )
        # No rngs stored: dropout keys are passed explicitly at call time so
        # the module state is pure parameters (simplifies jit/checkpointing).
        self.attn_dropout = nnx.Dropout(config.attention_dropout_rate)

    def __call__(
        self,
        x: jax.Array,
        *,
        cache: LayerCache | None = None,
        cache_length: jax.Array | None = None,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jax.Array, LayerCache | None]:
        """Attend over ``x`` of shape ``[batch, s, hidden]``.

        Args:
            cache: fixed-capacity layer cache, or None for pure training mode.
            cache_length: scalar count of valid cached positions (required
                with ``cache``); the new tokens occupy positions
                ``cache_length .. cache_length + s - 1``.
            deterministic: disables attention dropout (evaluation/inference).

        Returns:
            ``(output [batch, s, hidden], updated cache or None)``.
        """
        if (cache is None) != (cache_length is None):
            raise ValueError("cache and cache_length must be provided together")
        batch, s, _ = x.shape
        offset = jnp.zeros((), jnp.int32) if cache_length is None else cache_length

        q = self.q_proj(x).reshape(batch, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, s, self.num_kv_heads, self.head_dim)

        # Rotate queries and the *new* keys at their absolute positions.
        positions = offset + jnp.arange(s, dtype=jnp.int32)  # [s]
        cos, sin = rope_angles(positions[None, :], self.head_dim, self.rope_theta)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is None:
            keys, values = k, v
            key_positions = positions  # [s]
            valid_length = None
            new_cache = None
        else:
            new_cache = update_layer(cache, k, v, offset)
            keys, values = new_cache.k, new_cache.v  # [batch, capacity, kv, hd]
            key_positions = jnp.arange(keys.shape[1], dtype=jnp.int32)
            valid_length = offset + s

        # Grouped-query attention: repeat KV heads up to the query head count.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            keys = jnp.repeat(keys, repeat, axis=2)
            values = jnp.repeat(values, repeat, axis=2)

        # [batch, heads, q, k] logits in float32 for a stable softmax.
        scale = self.head_dim**-0.5
        logits = (
            jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), keys.astype(jnp.float32)) * scale
        )

        causal = key_positions[None, :] <= positions[:, None]  # [q, k]
        mask = causal
        if valid_length is not None:
            mask = mask & (key_positions[None, :] < valid_length)
        logits = jnp.where(mask[None, None, :, :], logits, _MASK_VALUE)

        probs = jax.nn.softmax(logits, axis=-1)
        probs = self.attn_dropout(probs, deterministic=deterministic, rngs=rngs)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs.astype(values.dtype), values)
        out = out.reshape(batch, s, self.num_heads * self.head_dim)
        return self.o_proj(out), new_cache
