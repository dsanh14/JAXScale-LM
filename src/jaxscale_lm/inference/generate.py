"""Generation loops: KV-cached and naive, with EOS handling and timing.

The loops are host-side Python driving jitted single-step functions — the
standard structure for autoregressive decoding with fixed shapes. Timing is
measured around synchronized regions (``block_until_ready``) so prefill and
decode latency are honest device-complete numbers.

EOS semantics: a per-row ``done`` mask is maintained on device; finished rows
emit ``pad_id``. The host checks the mask each step and stops early once all
rows are done (this is a deliberate host-device sync per step; see
docs/benchmarking.md for its effect on decode timing).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxscale_lm.inference.decode import CachedDecodeFn, NaiveDecodeFn
from jaxscale_lm.inference.prefill import PrefillFn
from jaxscale_lm.inference.sampling import SamplingParams, select_token
from jaxscale_lm.model.cache import KVCache
from jaxscale_lm.types import PRNGKey


@dataclass(frozen=True)
class GenerationTiming:
    """Wall-clock seconds for the two generation phases (synchronized)."""

    prefill_s: float
    decode_s: float
    decode_steps: int

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s


@dataclass(frozen=True)
class GenerationOutput:
    """Generated ids (EOS row-trimmed by the caller via pad_id) + timing."""

    token_ids: np.ndarray  # [batch, steps] int32, pad_id after a row's EOS
    timing: GenerationTiming


def _seen_mask(prompt_ids: jax.Array, vocab_size: int) -> jax.Array:
    """[batch, vocab] bool mask of tokens present in the prompt."""
    batch = prompt_ids.shape[0]
    mask = jnp.zeros((batch, vocab_size), dtype=bool)
    return mask.at[jnp.arange(batch)[:, None], prompt_ids].set(True)


def cached_generate(
    prefill_fn: PrefillFn,
    decode_fn: CachedDecodeFn,
    params,
    prompt_ids: jax.Array,
    cache: KVCache,
    *,
    max_new_tokens: int,
    sampling: SamplingParams,
    key: PRNGKey,
    eos_id: int | None,
    pad_id: int,
    vocab_size: int,
) -> GenerationOutput:
    """Prefill the prompt, then decode one token at a time using the cache."""
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    start = time.perf_counter()
    logits, cache = prefill_fn(params, prompt_ids, cache)
    jax.block_until_ready(logits)
    prefill_s = time.perf_counter() - start

    batch = prompt_ids.shape[0]
    done = jnp.zeros((batch,), dtype=bool)
    seen = _seen_mask(prompt_ids, vocab_size)
    tokens: list[jax.Array] = []

    start = time.perf_counter()
    steps = 0
    for step in range(max_new_tokens):
        token = select_token(logits, sampling, jax.random.fold_in(key, step), seen)
        if eos_id is not None:
            token = jnp.where(done, pad_id, token)
            done = done | (token == eos_id)
        seen = seen.at[jnp.arange(batch), token].set(True)
        tokens.append(token)
        steps += 1
        if eos_id is not None and bool(done.all()):
            break
        if step < max_new_tokens - 1:
            logits, cache = decode_fn(params, token[:, None], cache)
    jax.block_until_ready(tokens[-1])
    decode_s = time.perf_counter() - start

    ids = np.asarray(jnp.stack(tokens, axis=1))
    return GenerationOutput(
        token_ids=ids,
        timing=GenerationTiming(prefill_s=prefill_s, decode_s=decode_s, decode_steps=steps),
    )


def naive_generate(
    naive_fn: NaiveDecodeFn,
    params,
    prompt_ids: jax.Array,
    *,
    capacity: int,
    max_new_tokens: int,
    sampling: SamplingParams,
    key: PRNGKey,
    eos_id: int | None,
    pad_id: int,
    vocab_size: int,
) -> GenerationOutput:
    """Generate by recomputing the full prefix every step (no cache).

    The prompt lives in a fixed ``[batch, capacity]`` buffer so the step
    function compiles once; causality guarantees the unused tail cannot
    influence the logits that are read.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    batch, prompt_len = prompt_ids.shape
    if prompt_len + max_new_tokens > capacity:
        raise ValueError(
            f"prompt_len ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"buffer capacity ({capacity})."
        )

    buffer = jnp.full((batch, capacity), pad_id, dtype=jnp.int32)
    buffer = jax.lax.dynamic_update_slice_in_dim(buffer, prompt_ids.astype(jnp.int32), 0, axis=1)

    # "Prefill-equivalent": the first full forward over the prompt.
    start = time.perf_counter()
    logits = naive_fn(params, buffer, jnp.asarray(prompt_len, jnp.int32))
    jax.block_until_ready(logits)
    prefill_s = time.perf_counter() - start

    done = jnp.zeros((batch,), dtype=bool)
    seen = _seen_mask(prompt_ids, vocab_size)
    tokens: list[jax.Array] = []

    start = time.perf_counter()
    steps = 0
    for step in range(max_new_tokens):
        token = select_token(logits, sampling, jax.random.fold_in(key, step), seen)
        if eos_id is not None:
            token = jnp.where(done, pad_id, token)
            done = done | (token == eos_id)
        seen = seen.at[jnp.arange(batch), token].set(True)
        tokens.append(token)
        steps += 1
        if eos_id is not None and bool(done.all()):
            break
        if step < max_new_tokens - 1:
            length = prompt_len + step + 1
            buffer = jax.lax.dynamic_update_slice_in_dim(
                buffer, token[:, None], prompt_len + step, axis=1
            )
            logits = naive_fn(params, buffer, jnp.asarray(length, jnp.int32))
    jax.block_until_ready(tokens[-1])
    decode_s = time.perf_counter() - start

    ids = np.asarray(jnp.stack(tokens, axis=1))
    return GenerationOutput(
        token_ids=ids,
        timing=GenerationTiming(prefill_s=prefill_s, decode_s=decode_s, decode_steps=steps),
    )
