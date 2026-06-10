"""Prompt prefill: process the whole prompt once, populating the KV cache.

Prefill is compute-bound (one big batched matmul pass over the prompt) while
decode is memory-bound (one token per step reading the whole cache). They are
compiled separately so each gets its own fixed shapes, and benchmarked
separately because their performance characteristics differ fundamentally.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx

from jaxscale_lm.model.cache import KVCache

PrefillFn = Callable[[nnx.State, jax.Array, KVCache], tuple[jax.Array, KVCache]]


def make_prefill_fn(graphdef: nnx.GraphDef) -> PrefillFn:
    """Build the prefill function (jit it at the call site).

    The returned function maps ``(params, prompt_ids [B, P], empty cache)``
    to ``(last-position logits [B, V], populated cache)``.
    """

    def prefill(
        params: nnx.State, prompt_ids: jax.Array, cache: KVCache
    ) -> tuple[jax.Array, KVCache]:
        model = nnx.merge(graphdef, params)
        logits, new_cache = model(prompt_ids, cache=cache, deterministic=True)
        assert new_cache is not None
        return logits[:, -1, :], new_cache

    return prefill
