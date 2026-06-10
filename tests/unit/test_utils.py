"""Tests for seeding, timing statistics, and pytree helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxscale_lm.utils.seed import fold_in, host_rng, make_key
from jaxscale_lm.utils.timing import TimingResult, measure
from jaxscale_lm.utils.tree import count_params, dtype_from_name, tree_allclose, tree_bytes

pytestmark = pytest.mark.unit


class TestSeed:
    def test_host_rng_deterministic(self):
        a = host_rng(0).integers(0, 1000, size=16)
        b = host_rng(0).integers(0, 1000, size=16)
        np.testing.assert_array_equal(a, b)

    def test_host_rng_streams_independent(self):
        a = host_rng(0, "data").integers(0, 1000, size=16)
        b = host_rng(0, "eval").integers(0, 1000, size=16)
        assert not np.array_equal(a, b)

    def test_fold_in_changes_key(self):
        import jax

        key = make_key(0)
        assert not np.array_equal(
            np.asarray(jax.random.key_data(fold_in(key, 1))),
            np.asarray(jax.random.key_data(fold_in(key, 2))),
        )


class TestTimingStats:
    def test_percentiles(self):
        r = TimingResult(samples_s=tuple(float(i) for i in range(1, 101)), warmup_iterations=0)
        assert r.percentile_s(50) == pytest.approx(50.5)
        assert r.percentile_s(0) == 1.0
        assert r.percentile_s(100) == 100.0
        assert r.median_s == pytest.approx(50.5)

    def test_percentile_range_validated(self):
        r = TimingResult(samples_s=(1.0,), warmup_iterations=0)
        with pytest.raises(ValueError, match="percentile"):
            r.percentile_s(101)

    def test_measure_runs_and_counts(self):
        calls = 0

        def fn():
            nonlocal calls
            calls += 1
            return jnp.zeros(4)

        result = measure(fn, warmup=2, iterations=5)
        assert calls == 7
        assert len(result.samples_s) == 5
        assert all(s >= 0 for s in result.samples_s)

    def test_measure_rejects_zero_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            measure(lambda: None, warmup=0, iterations=0)


class TestTree:
    def test_count_params(self):
        tree = {"a": jnp.zeros((2, 3)), "b": [jnp.zeros(5), jnp.zeros(())]}
        assert count_params(tree) == 6 + 5 + 1

    def test_tree_bytes_respects_dtype(self):
        tree = {"a": jnp.zeros((4,), dtype=jnp.float32), "b": jnp.zeros((4,), dtype=jnp.bfloat16)}
        assert tree_bytes(tree) == 4 * 4 + 4 * 2

    def test_dtype_mapping(self):
        assert dtype_from_name("bfloat16") == jnp.bfloat16
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_from_name("int8")  # type: ignore[arg-type]

    def test_tree_allclose(self):
        a = {"x": jnp.ones(3)}
        assert tree_allclose(a, {"x": jnp.ones(3) + 1e-8})
        assert not tree_allclose(a, {"x": jnp.zeros(3)})
        assert not tree_allclose(a, {"y": jnp.ones(3)})
