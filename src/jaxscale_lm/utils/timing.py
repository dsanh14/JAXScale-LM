"""Timing utilities that are honest about JAX's asynchronous dispatch.

JAX dispatches work to devices asynchronously: a Python-level timer around a
jitted call measures *dispatch* time, not *execution* time, unless the result
is explicitly synchronized. Every timed region here ends with
``jax.block_until_ready`` on the function outputs.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class TimingResult:
    """Wall-clock samples (seconds) for a synchronized callable."""

    samples_s: tuple[float, ...]
    warmup_iterations: int

    @property
    def mean_s(self) -> float:
        return statistics.fmean(self.samples_s)

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s)

    @property
    def std_s(self) -> float:
        return statistics.stdev(self.samples_s) if len(self.samples_s) > 1 else 0.0

    def percentile_s(self, q: float) -> float:
        """Linear-interpolated percentile, q in [0, 100]."""
        if not 0 <= q <= 100:
            raise ValueError(f"percentile q must be in [0, 100], got {q}")
        data = sorted(self.samples_s)
        if len(data) == 1:
            return data[0]
        pos = (len(data) - 1) * q / 100
        lo = int(pos)
        hi = min(lo + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (pos - lo)


# XLA backend-compilation event key emitted by jax (jax._src.dispatch).
_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_compile_count = 0
_listener_registered = False


def _on_event_duration(event: str, duration: float, **_: Any) -> None:
    global _compile_count
    if event == _BACKEND_COMPILE_EVENT:
        _compile_count += 1


def compilation_count() -> int:
    """Global count of XLA backend compilations observed in this process.

    Counting starts at the first call (the listener is registered lazily),
    so call it once *before* the region you want to watch; an unexpected
    increase between two reads means something recompiled — typically a
    shape or static-argument change. This is process-global, not
    per-function: JAX no longer exposes per-jit cache sizes publicly.
    """
    global _listener_registered
    if not _listener_registered:
        jax.monitoring.register_event_duration_secs_listener(_on_event_duration)
        _listener_registered = True
    return _compile_count


def time_synchronized(fn: Callable[[], Any]) -> float:
    """Run ``fn`` once and return wall seconds including device completion."""
    start = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - start


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    iterations: int = 10,
) -> TimingResult:
    """Measure steady-state latency of ``fn``.

    Warmup runs (which absorb compilation) are executed and synchronized but
    not recorded. Use :func:`time_synchronized` separately for the first-call
    (compile + execute) measurement, *before* any warmup of the same function.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = tuple(time_synchronized(fn) for _ in range(iterations))
    return TimingResult(samples_s=samples, warmup_iterations=warmup)
