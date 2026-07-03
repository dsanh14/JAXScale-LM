"""Environment and git-state capture shared by benchmarks and run manifests.

Everything here is host-side introspection: safe to call anywhere outside
jitted code, never fatal if git is unavailable (recorded as unknown instead).
"""

from __future__ import annotations

import platform as platform_mod
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Not a git repo / git unavailable: recorded as unknown, not fatal.
        return None


def git_info() -> dict[str, Any]:
    """Capture git commit, dirty state, and branch (None when unavailable)."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def runtime_info() -> dict[str, Any]:
    """Capture Python/JAX/library versions and the device topology."""
    from importlib.metadata import version as pkg_version

    import flax
    import jax
    import optax
    import orbax.checkpoint as ocp

    devices = jax.devices()
    return {
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "jaxlib_version": pkg_version("jaxlib"),
        "flax_version": flax.__version__,
        "optax_version": optax.__version__,
        "orbax_version": ocp.__version__,
        "platform": jax.default_backend(),
        "host": platform_mod.platform(),
        "device_names": sorted({d.device_kind for d in devices}),
        "device_count": jax.device_count(),
        "process_count": jax.process_count(),
    }


def environment_info() -> dict[str, Any]:
    """Combined git + runtime capture (the benchmark-record ``environment``).

    Key set and semantics are part of benchmark schema v1 for the original
    fields; ``git_branch`` was added later and downstream tooling must not
    require it.
    """
    return {**git_info(), **runtime_info()}


def new_run_id() -> str:
    """Sortable, collision-resistant run identifier."""
    return f"{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
