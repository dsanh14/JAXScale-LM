"""Token sampling: greedy, temperature, top-k, top-p, repetition penalty.

All filters operate on float32 logits of shape ``[batch, vocab]`` and run
inside jit. Static knobs (k, p, flags) are baked into the compiled function
via :class:`SamplingParams` (hashable frozen dataclass) — changing them
recompiles; changing the seed or batch contents does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

_FILTER_VALUE = -1e30


@dataclass(frozen=True)
class SamplingParams:
    """Static sampling configuration (hashable; safe as a jit static arg)."""

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    repetition_penalty: float | None = None

    def validate(self, vocab_size: int) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0 when sampling, got {self.temperature}")
        if self.top_k < 0 or self.top_k > vocab_size:
            raise ValueError(f"top_k must be in [0, vocab_size={vocab_size}], got {self.top_k}")
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")


def top_k_filter(logits: jax.Array, k: int) -> jax.Array:
    """Keep the k highest logits per row; mask the rest."""
    if k <= 0:
        return logits
    threshold = jax.lax.top_k(logits, k)[0][..., -1:]
    return jnp.where(logits < threshold, _FILTER_VALUE, logits)


def top_p_filter(logits: jax.Array, p: float) -> jax.Array:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability >= p (the token that crosses the boundary is kept)."""
    if p >= 1.0:
        return logits
    sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
    cumprobs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    # Position of each sorted token relative to the nucleus boundary; shift
    # by one so the crossing token stays included.
    sorted_keep = cumprobs - jax.nn.softmax(sorted_logits, axis=-1) < p
    # Map back: a logit is kept iff it is >= the smallest kept sorted logit.
    kept_count = jnp.maximum(sorted_keep.sum(axis=-1, keepdims=True), 1)
    cutoff = jnp.take_along_axis(sorted_logits, kept_count - 1, axis=-1)
    return jnp.where(logits < cutoff, _FILTER_VALUE, logits)


def apply_repetition_penalty(logits: jax.Array, seen_mask: jax.Array, penalty: float) -> jax.Array:
    """Penalize tokens already generated (CTRL-style).

    Args:
        seen_mask: ``[batch, vocab]`` bool, True for tokens present in the
            output so far.
    """
    penalized = jnp.where(logits > 0, logits / penalty, logits * penalty)
    return jnp.where(seen_mask, penalized, logits)


def select_token(
    logits: jax.Array,
    params: SamplingParams,
    key: jax.Array,
    seen_mask: jax.Array | None = None,
) -> jax.Array:
    """Pick the next token id per row from ``[batch, vocab]`` logits."""
    logits = logits.astype(jnp.float32)
    if params.repetition_penalty is not None and seen_mask is not None:
        logits = apply_repetition_penalty(logits, seen_mask, params.repetition_penalty)
    if not params.do_sample:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    logits = logits / params.temperature
    logits = top_k_filter(logits, params.top_k)
    logits = top_p_filter(logits, params.top_p)
    return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
