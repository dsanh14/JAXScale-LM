"""Pre-norm decoder block: norm -> attention -> residual, norm -> MLP -> residual."""

from __future__ import annotations

import jax
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.model.attention import CausalSelfAttention
from jaxscale_lm.model.cache import LayerCache
from jaxscale_lm.model.mlp import MLP
from jaxscale_lm.utils.tree import dtype_from_name


class DecoderBlock(nnx.Module):
    """One Transformer decoder block (pre-normalization, RMSNorm)."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        param_dtype = dtype_from_name(config.parameter_dtype)
        compute_dtype = dtype_from_name(config.compute_dtype)
        self.attn_norm = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.normalization_epsilon,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.attention = CausalSelfAttention(config, rngs)
        self.mlp_norm = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.normalization_epsilon,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.mlp = MLP(config, rngs)
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self,
        x: jax.Array,
        *,
        cache: LayerCache | None = None,
        cache_length: jax.Array | None = None,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[jax.Array, LayerCache | None]:
        attn_out, new_cache = self.attention(
            self.attn_norm(x),
            cache=cache,
            cache_length=cache_length,
            deterministic=deterministic,
            rngs=rngs,
        )
        x = x + self.dropout(attn_out, deterministic=deterministic, rngs=rngs)
        x = x + self.mlp(self.mlp_norm(x), deterministic=deterministic, rngs=rngs)
        return x, new_cache
