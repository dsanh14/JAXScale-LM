"""Next-token cross-entropy loss and per-batch sufficient statistics.

Losses are computed in float32 regardless of the model's compute dtype
(the model already returns float32 logits) and reduced as *sums* plus a
valid-token count rather than means. Sums are the right primitive because:

- gradient accumulation needs `Σ nll / Σ tokens` across microbatches to
  match the equivalent large-batch update exactly, and
- evaluation must aggregate token-weighted across batches of unequal size
  (averaging per-batch means would be wrong).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from jaxscale_lm.types import ArrayLike


class LossStats(NamedTuple):
    """Sufficient statistics for token-weighted aggregation (all float32)."""

    nll_sum: jax.Array  # Σ negative log-likelihood over valid tokens
    correct_sum: jax.Array  # Σ argmax(logits) == target over valid tokens
    valid_tokens: jax.Array  # Σ loss_mask


def loss_stats(logits: jax.Array, target_ids: ArrayLike, loss_mask: ArrayLike) -> LossStats:
    """Compute summed NLL / accuracy statistics for one (micro)batch.

    Args:
        logits: ``[batch, seq, vocab]`` float32.
        target_ids: ``[batch, seq]`` int32 (device or host array).
        loss_mask: ``[batch, seq]`` float32, 1.0 on real targets.
    """
    target_ids = jnp.asarray(target_ids)
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    mask = jnp.asarray(loss_mask, jnp.float32)
    correct = (jnp.argmax(logits, axis=-1) == target_ids).astype(jnp.float32)
    return LossStats(
        nll_sum=jnp.sum(nll * mask),
        correct_sum=jnp.sum(correct * mask),
        valid_tokens=jnp.sum(mask),
    )


def mean_loss(stats: LossStats) -> jax.Array:
    """Token-weighted mean cross-entropy."""
    return stats.nll_sum / jnp.maximum(stats.valid_tokens, 1.0)


def perplexity(mean_nll: float) -> float:
    """Perplexity from a mean NLL (host-side; exp of large values saturates)."""
    return float(jnp.exp(jnp.minimum(jnp.asarray(mean_nll), 80.0)))
