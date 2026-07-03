"""Per-run reproducibility manifest.

Every training run writes a self-contained audit trail under
``<artifacts_dir>/runs/<run_id>/``:

```
resolved_config.yaml   exact configuration the run executed with
environment.json       Python/JAX/library versions, backend, devices
git.json               commit, dirty flag, branch (null outside a repo)
run.json               run id/name, command line, checkpoint directory, seed
metrics.jsonl          one JSON object per logged event (train/eval/final)
checkpoints            symlink to the stable checkpoint directory
```

Checkpoints deliberately live *outside* the run directory (at the stable
``<artifacts_dir>/checkpoints/<run_name>``) because resumption must find
them across invocations; each invocation is its own run with its own
manifest, and the symlink + ``run.json`` record the linkage.

All writes are plain host-side file I/O — never called from jitted code.
"""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jaxscale_lm.utils.environment import git_info, new_run_id, runtime_info

if TYPE_CHECKING:
    from jaxscale_lm.config import Config


class RunManifest:
    """Writes the reproducibility record for one run invocation.

    Create via :meth:`create`, append events with :meth:`log_metrics`.
    Metric writes are appended and flushed line-by-line so a crashed run
    still leaves a valid (truncated) ``metrics.jsonl``.
    """

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self._metrics_path = run_dir / "metrics.jsonl"

    @classmethod
    def create(cls, config: Config, *, checkpoint_dir: Path) -> RunManifest:
        """Create ``<artifacts_dir>/runs/<run_id>/`` and write static records."""
        from jaxscale_lm.config import save_resolved_config

        run_id = new_run_id()
        run_dir = config.project.artifacts_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        save_resolved_config(config, run_dir / "resolved_config.yaml")
        _write_json(run_dir / "environment.json", runtime_info())
        _write_json(run_dir / "git.json", git_info())
        _write_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "run_name": config.project.run_name,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "command": sys.argv,
                "seed": config.project.seed,
                "checkpoint_dir": str(checkpoint_dir),
            },
        )

        # Filesystems without symlink support still get the checkpoint path
        # via run.json; the manifest stays complete.
        link = run_dir / "checkpoints"
        with contextlib.suppress(OSError):
            link.symlink_to(checkpoint_dir.resolve(), target_is_directory=True)

        return cls(run_dir, run_id)

    def log_metrics(self, event: str, step: int | None = None, **values: Any) -> None:
        """Append one metrics event as a JSON line (immediately durable)."""
        payload: dict[str, Any] = {
            "event": event,
            "step": step,
            "time": datetime.now(tz=UTC).isoformat(),
            **values,
        }
        with self._metrics_path.open("a") as f:
            f.write(json.dumps(payload, default=_jsonable) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
        f.write("\n")


def _jsonable(value: Any) -> Any:
    """Fallback encoder: numpy/JAX scalars and paths become plain JSON types."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)
