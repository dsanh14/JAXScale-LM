"""The decoder-only Transformer language model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.model.block import DecoderBlock
from jaxscale_lm.model.cache import KVCache
from jaxscale_lm.model.embeddings import TokenEmbedding
from jaxscale_lm.utils.tree import count_params, dtype_from_name


class Transformer(nnx.Module):
    """Token embedding -> N pre-norm decoder blocks -> final norm -> LM head.

    Logits are always returned in float32: the final projection feeds either
    a cross-entropy loss or a sampling softmax, both of which are numerically
    sensitive reductions.
    """

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        self.config = config
        param_dtype = dtype_from_name(config.parameter_dtype)
        compute_dtype = dtype_from_name(config.compute_dtype)
        self.embed = TokenEmbedding(config, rngs)
        self.blocks = nnx.List([DecoderBlock(config, rngs) for _ in range(config.num_layers)])
        self.final_norm = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.normalization_epsilon,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            rngs=rngs,
        )
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                kernel_init=nnx.initializers.normal(config.initializer_range),
                param_dtype=param_dtype,
                dtype=compute_dtype,
                rngs=rngs,
            )
        self.embed_dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        cache: KVCache | None = None,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jax.Array, KVCache | None]:
        """Run the model over ``input_ids`` of shape ``[batch, s]``.

        Without a cache this is the full-sequence training/eval path. With a
        cache, the ``s`` new tokens are placed at positions
        ``cache.length .. cache.length + s - 1`` (prefill when ``length`` is
        0, single-token decode when ``s`` is 1) and an updated cache is
        returned.

        Returns:
            ``(logits [batch, s, vocab] float32, updated KVCache or None)``.
        """
        x = self.embed(input_ids)
        x = self.embed_dropout(x, deterministic=deterministic, rngs=rngs)

        new_layers = []
        cache_length = cache.length if cache is not None else None
        for i, block in enumerate(self.blocks):
            layer_cache = cache.layers[i] if cache is not None else None
            x, new_layer = block(
                x,
                cache=layer_cache,
                cache_length=cache_length,
                deterministic=deterministic,
                rngs=rngs,
            )
            if new_layer is not None:
                new_layers.append(new_layer)

        x = self.final_norm(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = self.embed.attend(x)
        logits = logits.astype(jnp.float32)

        new_cache = None
        if cache is not None:
            new_cache = KVCache(
                layers=tuple(new_layers),
                length=cache.length + jnp.asarray(input_ids.shape[1], jnp.int32),
            )
        return logits, new_cache

    def num_params(self) -> int:
        """Number of trainable parameters (actual, not estimated)."""
        return count_params(nnx.state(self, nnx.Param))


def build_model(config: ModelConfig, seed: int) -> Transformer:
    """Construct and initialize a Transformer from config + seed."""
    return Transformer(config, nnx.Rngs(seed))
