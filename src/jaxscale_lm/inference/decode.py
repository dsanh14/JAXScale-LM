"""Single-token decode steps: cached and naive.

Cached decode: O(cache) attention per token, fixed shapes, compiles once.

Naive decode: recomputes the *entire* prefix every step. To keep it honest
but still compile-once, the prefix lives in a fixed ``[B, capacity]`` buffer:
causal attention guarantees positions past the current length cannot affect
the logits we read, so the buffer tail can hold arbitrary ids. The
per-step cost is a full O(capacity²) forward pass — exactly the complexity
KV caching removes.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx

from jaxscale_lm.model.cache import KVCache

CachedDecodeFn = Callable[[nnx.State, jax.Array, KVCache], tuple[jax.Array, KVCache]]
NaiveDecodeFn = Callable[[nnx.State, jax.Array, jax.Array], jax.Array]


def make_cached_decode_fn(graphdef: nnx.GraphDef) -> CachedDecodeFn:
    """Build the KV-cached decode step (jit at the call site).

    Maps ``(params, token [B, 1], cache)`` to ``(logits [B, V], cache)``.
    The token is written at position ``cache.length`` via an indexed update;
    nothing else is recomputed.
    """

    def decode(
        params: nnx.State, token: jax.Array, cache: KVCache
    ) -> tuple[jax.Array, KVCache]:
        model = nnx.merge(graphdef, params)
        logits, new_cache = model(token, cache=cache, deterministic=True)
        assert new_cache is not None
        return logits[:, -1, :], new_cache

    return decode


def make_naive_decode_fn(graphdef: nnx.GraphDef) -> NaiveDecodeFn:
    """Build the naive (full-prefix recompute) decode step.

    Maps ``(params, buffer [B, capacity], length scalar)`` to next-token
    logits ``[B, V]`` taken at position ``length - 1``.
    """

    def decode(params: nnx.State, buffer: jax.Array, length: jax.Array) -> jax.Array:
        model = nnx.merge(graphdef, params)
        logits, _ = model(buffer, deterministic=True)
        # Gather the logits at the last real position of each row.
        return jax.lax.dynamic_index_in_dim(logits, length - 1, axis=1, keepdims=False)

    return decode
