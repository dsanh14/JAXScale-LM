"""Fixed-capacity KV cache.

Layout decision
---------------
Keys and values are stored as ``[batch, capacity, num_kv_heads, head_dim]``:

- ``capacity`` (the maximum sequence length for this generation) is a
  *static* dimension, so every decode step has identical shapes and XLA
  compiles the step exactly once.
- Sequence-major layout (position as axis 1) lets prefill and decode write
  with a single ``jax.lax.dynamic_update_slice_in_dim`` along one axis, and
  matches the ``[batch, seq, heads, head_dim]`` activation layout used
  elsewhere, avoiding transposes on the hot path.

``length`` is a traced scalar (the number of valid positions, shared by all
batch rows). Batched generation therefore assumes equal prompt lengths per
batch — a documented limitation (see docs/limitations.md); ragged batches
would need per-row lengths and key-side masking by row.

The cache is a NamedTuple, i.e. an ordinary JAX pytree: it flows in and out
of jitted functions with no special handling and updates are functional
(a new cache is returned, never mutated in place).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.utils.tree import dtype_from_name


class LayerCache(NamedTuple):
    """Cached keys/values for one decoder block (post-RoPE)."""

    k: jax.Array  # [batch, capacity, num_kv_heads, head_dim]
    v: jax.Array  # [batch, capacity, num_kv_heads, head_dim]


class KVCache(NamedTuple):
    """Per-layer caches plus the shared number of valid positions."""

    layers: tuple[LayerCache, ...]
    length: jax.Array  # scalar int32

    @property
    def capacity(self) -> int:
        return int(self.layers[0].k.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.layers[0].k.shape[0])


def init_cache(config: ModelConfig, batch_size: int, capacity: int) -> KVCache:
    """Allocate an empty cache.

    Raises:
        ValueError: if capacity exceeds the model's maximum sequence length
            (RoPE positions beyond it were never seen in training) or is
            non-positive.
    """
    if capacity <= 0:
        raise ValueError(f"KV cache capacity must be positive, got {capacity}")
    if capacity > config.max_sequence_length:
        raise ValueError(
            f"KV cache capacity {capacity} exceeds model.max_sequence_length "
            f"{config.max_sequence_length}; generation cannot run past the "
            f"trained context window."
        )
    if batch_size <= 0:
        raise ValueError(f"KV cache batch_size must be positive, got {batch_size}")
    shape = (batch_size, capacity, config.kv_heads, config.head_dim)
    dtype = dtype_from_name(config.compute_dtype)
    layers = tuple(
        LayerCache(k=jnp.zeros(shape, dtype), v=jnp.zeros(shape, dtype))
        for _ in range(config.num_layers)
    )
    return KVCache(layers=layers, length=jnp.zeros((), jnp.int32))


def update_layer(
    layer: LayerCache, k_new: jax.Array, v_new: jax.Array, offset: jax.Array
) -> LayerCache:
    """Write ``k_new``/``v_new`` (``[batch, s, kv_heads, head_dim]``) at ``offset``.

    Uses ``dynamic_update_slice_in_dim`` along the capacity axis — a fixed-
    shape indexed write, never a concatenation of the history.
    """
    k = jax.lax.dynamic_update_slice_in_dim(layer.k, k_new.astype(layer.k.dtype), offset, axis=1)
    v = jax.lax.dynamic_update_slice_in_dim(layer.v, v_new.astype(layer.v.dtype), offset, axis=1)
    return LayerCache(k=k, v=v)
