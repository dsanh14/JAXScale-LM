"""Explicit, reproducible randomness.

All stochastic behavior in JAXScale-LM flows from a single integer seed via
JAX PRNG keys. Host-side shuffling uses an independent ``numpy`` generator
derived from the same seed so device and host randomness never interact.
"""

from __future__ import annotations

import zlib

import jax
import numpy as np

from jaxscale_lm.types import PRNGKey


def make_key(seed: int) -> PRNGKey:
    """Create the root JAX PRNG key for a run."""
    return jax.random.key(seed)


def fold_in(key: PRNGKey, data: int) -> PRNGKey:
    """Derive a stream-specific key (e.g. per-step) without consuming the parent."""
    return jax.random.fold_in(key, data)


def host_rng(seed: int, stream: str = "data") -> np.random.Generator:
    """Deterministic numpy generator for host-side work (shuffling, synthetic data)."""
    # zlib.crc32 is stable across processes (unlike built-in str hashing,
    # which is randomized per interpreter and would break reproducibility).
    ss = np.random.SeedSequence([seed, zlib.crc32(stream.encode())])
    return np.random.default_rng(ss)
