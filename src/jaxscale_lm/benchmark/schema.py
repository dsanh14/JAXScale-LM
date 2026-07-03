"""Versioned benchmark records.

Every record is self-describing: environment (git, library versions,
devices), workload parameters, raw samples, and derived statistics. Raw
samples are always preserved so statistics can be recomputed and runs can be
compared post-hoc. ``schema_version`` gates downstream tooling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from jaxscale_lm.utils.environment import environment_info, new_run_id
from jaxscale_lm.utils.timing import TimingResult

__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkRecord",
    "environment_info",
    "new_run_id",
    "record_from_timing",
]

SCHEMA_VERSION = 1


@dataclass
class BenchmarkRecord:
    """One benchmark measurement (or one recorded failure)."""

    suite: str
    name: str
    mode: str  # e.g. "steady_state", "first_call", "eager"
    status: str = "ok"  # "ok" | "failed"
    error: str | None = None

    # Workload parameters (None = not applicable to this workload).
    parameter_count: int | None = None
    dtype: str | None = None
    batch_size: int | None = None
    prompt_length: int | None = None
    generate_length: int | None = None
    sequence_length: int | None = None
    seed: int | None = None
    warmup_iterations: int | None = None
    measure_iterations: int | None = None

    # Measurements.
    raw_samples_s: list[float] = field(default_factory=list)
    mean_s: float | None = None
    median_s: float | None = None
    std_s: float | None = None
    p50_s: float | None = None
    p90_s: float | None = None
    p95_s: float | None = None
    p99_s: float | None = None

    # Derived workload-specific numbers (tokens/s, ms/token, ...).
    extra: dict[str, Any] = field(default_factory=dict)

    # Run context (filled by the runner).
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    timestamp: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_from_timing(
    suite: str,
    name: str,
    mode: str,
    timing: TimingResult,
    **fields: Any,
) -> BenchmarkRecord:
    """Build a record from synchronized timing samples."""
    n = len(timing.samples_s)
    return BenchmarkRecord(
        suite=suite,
        name=name,
        mode=mode,
        raw_samples_s=list(timing.samples_s),
        mean_s=timing.mean_s,
        median_s=timing.median_s,
        std_s=timing.std_s,
        p50_s=timing.percentile_s(50),
        p90_s=timing.percentile_s(90),
        p95_s=timing.percentile_s(95),
        # p99 is meaningless with few samples; require >= 20.
        p99_s=timing.percentile_s(99) if n >= 20 else None,
        warmup_iterations=timing.warmup_iterations,
        measure_iterations=n,
        **fields,
    )
