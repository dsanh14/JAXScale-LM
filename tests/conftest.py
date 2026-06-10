"""Shared test fixtures.

Tests run on CPU regardless of the host's accelerators so results are
hermetic; accelerator-specific tests are marked and opt-in.
"""

from __future__ import annotations

import os

# Must happen before any test module imports jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from jaxscale_lm.config import Config, load_config

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture(scope="session")
def smoke_config() -> Config:
    """The CPU smoke-test config, exactly as `make train-smoke` sees it."""
    return load_config(os.path.join(_CONFIG_DIR, "train", "cpu_smoke.yaml"))


@pytest.fixture()
def tmp_artifacts(tmp_path, smoke_config: Config) -> Config:
    """Smoke config redirected to a per-test artifacts directory."""
    return smoke_config.model_copy(
        update={
            "project": smoke_config.project.model_copy(
                update={"artifacts_dir": tmp_path / "artifacts"}
            )
        }
    )
