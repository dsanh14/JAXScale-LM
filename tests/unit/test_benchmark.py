"""Benchmark harness tests: schema stability, statistics, memory probe."""

from __future__ import annotations

import pytest

from jaxscale_lm.benchmark.memory import snapshot
from jaxscale_lm.benchmark.schema import (
    SCHEMA_VERSION,
    BenchmarkRecord,
    environment_info,
    record_from_timing,
)
from jaxscale_lm.utils.timing import TimingResult

pytestmark = pytest.mark.unit


class TestSchema:
    def test_record_round_trips_to_dict(self):
        record = BenchmarkRecord(suite="s", name="n", mode="m")
        data = record.to_dict()
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["status"] == "ok"

    def test_record_from_timing_statistics(self):
        timing = TimingResult(
            samples_s=tuple(float(i) for i in range(1, 11)), warmup_iterations=2
        )
        record = record_from_timing("s", "n", "steady_state", timing, batch_size=4)
        assert record.mean_s == pytest.approx(5.5)
        assert record.median_s == pytest.approx(5.5)
        assert record.p90_s == pytest.approx(9.1)
        assert record.p99_s is None  # < 20 samples
        assert record.raw_samples_s == [float(i) for i in range(1, 11)]
        assert record.measure_iterations == 10
        assert record.batch_size == 4

    def test_p99_present_with_enough_samples(self):
        timing = TimingResult(samples_s=tuple(float(i) for i in range(100)), warmup_iterations=0)
        record = record_from_timing("s", "n", "m", timing)
        assert record.p99_s is not None

    def test_schema_fields_stable(self):
        """Regression: downstream tooling relies on these exact field names."""
        expected = {
            "suite",
            "name",
            "mode",
            "status",
            "error",
            "parameter_count",
            "dtype",
            "batch_size",
            "prompt_length",
            "generate_length",
            "sequence_length",
            "seed",
            "warmup_iterations",
            "measure_iterations",
            "raw_samples_s",
            "mean_s",
            "median_s",
            "std_s",
            "p50_s",
            "p90_s",
            "p95_s",
            "p99_s",
            "extra",
            "schema_version",
            "run_id",
            "timestamp",
            "environment",
            "model_config",
        }
        assert set(BenchmarkRecord(suite="s", name="n", mode="m").to_dict()) == expected


class TestEnvironmentInfo:
    def test_required_keys_present(self):
        info = environment_info()
        for key in (
            "git_commit",
            "git_dirty",
            "python_version",
            "jax_version",
            "jaxlib_version",
            "flax_version",
            "optax_version",
            "orbax_version",
            "platform",
            "device_names",
            "device_count",
            "process_count",
        ):
            assert key in info
        assert info["platform"] == "cpu"
        assert info["device_count"] >= 1


class TestMemory:
    def test_snapshot_honest_on_cpu(self):
        snap = snapshot()
        # Host RSS should be available via psutil; device stats on CPU must
        # be reported as unsupported, never invented.
        assert snap.host_rss_bytes is None or snap.host_rss_bytes > 0
        assert snap.device_stats == "unsupported" or isinstance(snap.device_stats, dict)
        data = snap.to_dict()
        assert "host_source" in data and "device_source" in data
