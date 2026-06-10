"""Position-wise feed-forward network (GELU MLP)."""

from __future__ import annotations

import jax
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.utils.tree import dtype_from_name


class MLP(nnx.Module):
    """``hidden -> intermediate -> GELU -> hidden`` with optional dropout."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        param_dtype = dtype_from_name(config.parameter_dtype)
        compute_dtype = dtype_from_name(config.compute_dtype)
        init = nnx.initializers.normal(config.initializer_range)
        self.up = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=config.use_bias,
            kernel_init=init,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=config.use_bias,
            kernel_init=init,
            param_dtype=param_dtype,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self, x: jax.Array, *, deterministic: bool = True, rngs: nnx.Rngs | None = None
    ) -> jax.Array:
        x = self.up(x)
        x = nnx.gelu(x)
        x = self.down(x)
        return self.dropout(x, deterministic=deterministic, rngs=rngs)
