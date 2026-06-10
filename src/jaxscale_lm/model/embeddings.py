"""Token embeddings and rotary positional embeddings (RoPE).

RoPE rotates query/key vectors pairwise by position-dependent angles, which
encodes *relative* positions in the attention dot product. Two properties
matter for this project:

- It is applied inside attention (per layer), so the KV cache stores
  *already-rotated* keys: a decode step only needs the rotation for its own
  position, never a recomputation of history.
- The rotation depends only on the absolute position index, so prefill
  (positions ``0..P-1``) and decode (position ``P+t``) use the same tables.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.utils.tree import dtype_from_name


class TokenEmbedding(nnx.Module):
    """Token-id -> vector table; also provides the tied output projection."""

    def __init__(self, config: ModelConfig, rngs: nnx.Rngs) -> None:
        self.embedding = nnx.Param(
            nnx.initializers.normal(config.initializer_range)(
                rngs.params(),
                (config.vocab_size, config.hidden_size),
                dtype_from_name(config.parameter_dtype),
            )
        )
        self.compute_dtype = dtype_from_name(config.compute_dtype)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """Embed ``[batch, seq]`` int ids to ``[batch, seq, hidden]``."""
        return self.embedding.value.astype(self.compute_dtype)[token_ids]

    def attend(self, hidden: jnp.ndarray) -> jnp.ndarray:
        """Tied output head: project hidden states onto the vocabulary."""
        return hidden @ self.embedding.value.astype(self.compute_dtype).T


def rope_angles(
    positions: jnp.ndarray, head_dim: int, theta: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cos/sin tables for the given absolute positions.

    Args:
        positions: ``[batch, seq]`` (or broadcastable) int positions.
        head_dim: per-head dimension (must be even).
        theta: RoPE base frequency.

    Returns:
        ``(cos, sin)`` each shaped ``positions.shape + (head_dim / 2,)``,
        computed in float32 for numerical stability.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    angles = positions.astype(jnp.float32)[..., None] * inv_freq
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Rotate ``x`` of shape ``[batch, seq, heads, head_dim]`` by (cos, sin).

    ``cos``/``sin`` are ``[batch, seq, head_dim/2]`` and broadcast over heads.
    The rotation is applied in float32 and cast back to the input dtype.
    """
    orig_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    x_even = x32[..., 0::2]
    x_odd = x32[..., 1::2]
    cos_b = cos[:, :, None, :]  # broadcast over heads
    sin_b = sin[:, :, None, :]
    rotated_even = x_even * cos_b - x_odd * sin_b
    rotated_odd = x_even * sin_b + x_odd * cos_b
    out = jnp.stack([rotated_even, rotated_odd], axis=-1).reshape(x.shape)
    return out.astype(orig_dtype)
