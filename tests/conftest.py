"""Shared test fixtures.

Tests run on CPU regardless of the host's accelerators so results are
hermetic; accelerator-specific tests are marked and opt-in.
"""

from __future__ import annotations

import os

# Must happen before any test module imports jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

try:
    from jaxscale_lm.config import Config, load_config
except ModuleNotFoundError as exc:  # pragma: no cover - environment failure path
    raise ModuleNotFoundError(
        "jaxscale_lm is not importable from the current environment. "
        "Most likely the venv holds a stale editable install whose .pth file "
        "was marked UF_HIDDEN by an external macOS process (Python >= 3.12.4 "
        "skips hidden .pth files). Fix: run `make install` (non-editable "
        "install, immune to this) or `make venv-fix`, then re-run the tests."
    ) from exc

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
