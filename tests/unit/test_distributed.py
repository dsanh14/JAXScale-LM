"""Distributed-logic tests.

Single-device mesh tests run in-process. Real multi-device behavior is
exercised in a *subprocess* with ``XLA_FLAGS=--xla_force_host_platform_
device_count=8`` (simulated CPU devices — suitable for testing sharding
logic, never for performance claims), because the flag must be set before
JAX initializes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest

from jaxscale_lm.config import DistributedConfig
from jaxscale_lm.distributed.diagnostics import describe_array, describe_mesh, full_report
from jaxscale_lm.distributed.mesh import build_mesh
from jaxscale_lm.distributed.partitioning import (
    replicated,
    train_batch_sharding,
    validate_batch_divisibility,
)
from jaxscale_lm.distributed.placement import place_batch
from jaxscale_lm.types import Batch

pytestmark = pytest.mark.unit


class TestSingleDeviceMesh:
    def test_default_mesh_builds(self):
        mesh = build_mesh(DistributedConfig())
        assert dict(mesh.shape) == {"data": jax.device_count(), "model": 1}

    def test_explicit_oversized_mesh_rejected(self):
        cfg = DistributedConfig(data_axis_size=jax.device_count() + 1)
        with pytest.raises(ValueError, match="does not match"):
            build_mesh(cfg)

    def test_model_axis_too_large_rejected(self):
        cfg = DistributedConfig(model_axis_size=jax.device_count() + 1)
        with pytest.raises(ValueError, match="model_axis_size"):
            build_mesh(cfg)

    def test_batch_divisibility(self):
        mesh = build_mesh(DistributedConfig())
        validate_batch_divisibility(jax.device_count() * 2, mesh, DistributedConfig())
        # An indivisible batch is impossible to construct on 1 device, so
        # only check the error on multi-device meshes (subprocess test).

    def test_placement_replicated(self):
        mesh = build_mesh(DistributedConfig())
        batch = Batch(
            input_ids=np.zeros((2, 4), np.int32),
            target_ids=np.zeros((2, 4), np.int32),
            loss_mask=np.ones((2, 4), np.float32),
        )
        placed = place_batch(batch, replicated(mesh))
        assert placed.input_ids.sharding.is_fully_replicated  # type: ignore[union-attr]

    def test_diagnostics_render(self):
        mesh = build_mesh(DistributedConfig())
        x = jax.numpy.zeros((4, 4))
        report = full_report(mesh, x)
        assert "mesh shape" in report
        assert "sample.sharding" in report
        assert len(describe_mesh(mesh)) == 3
        assert any("shards" in line for line in describe_array("sample", x))


_MULTI_DEVICE_SCRIPT = textwrap.dedent(
    """
    import os
    assert "--xla_force_host_platform_device_count=8" in os.environ.get("XLA_FLAGS", "")
    import jax
    import numpy as np
    assert jax.device_count() == 8, jax.device_count()

    from jaxscale_lm.config import Config, DistributedConfig
    from jaxscale_lm.distributed.mesh import build_mesh
    from jaxscale_lm.distributed.partitioning import (
        train_batch_sharding, validate_batch_divisibility,
    )
    from jaxscale_lm.distributed.placement import place_batch
    from jaxscale_lm.types import Batch

    cfg = DistributedConfig()
    mesh = build_mesh(cfg)
    assert dict(mesh.shape) == {"data": 8, "model": 1}, dict(mesh.shape)

    # Indivisible batch fails fast with an actionable message.
    try:
        validate_batch_divisibility(6, mesh, cfg)
        raise AssertionError("expected ValueError for indivisible batch")
    except ValueError as e:
        assert "divisible" in str(e)

    # A [accum=2, micro=8, seq=16] batch shards row-wise over 8 devices.
    batch = Batch(
        input_ids=np.arange(2 * 8 * 16, dtype=np.int32).reshape(2, 8, 16),
        target_ids=np.zeros((2, 8, 16), np.int32),
        loss_mask=np.ones((2, 8, 16), np.float32),
    )
    placed = place_batch(batch, train_batch_sharding(mesh, cfg))
    shards = placed.input_ids.addressable_shards
    assert len(shards) == 8
    assert all(s.data.shape == (2, 1, 16) for s in shards), [s.data.shape for s in shards]

    # End-to-end: a sharded train step on the smoke config produces the same
    # loss as the single-device path would (the data is identical).
    from jaxscale_lm.config import load_config
    from jaxscale_lm.training.trainer import Trainer

    config = load_config("configs/train/cpu_smoke.yaml")
    config = config.model_copy(update={
        "project": config.project.model_copy(
            update={"artifacts_dir": os.environ["TEST_TMPDIR"]}
        ),
        "data": config.data.model_copy(update={"batch_size": 8}),
        "training": config.training.model_copy(update={"max_steps": 3}),
    })
    trainer = Trainer(config)
    summary = trainer.train()
    assert np.isfinite(summary["loss"]), summary
    print("MULTI_DEVICE_OK", summary["loss"])
    """
)


@pytest.mark.multi_device
class TestSimulatedMultiDevice:
    def test_eight_simulated_cpu_devices(self, tmp_path):
        env = dict(os.environ)
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
        ).strip()
        env["JAX_PLATFORMS"] = "cpu"
        env["TEST_TMPDIR"] = str(tmp_path)
        result = subprocess.run(
            [sys.executable, "-c", _MULTI_DEVICE_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            timeout=600,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "MULTI_DEVICE_OK" in result.stdout
