"""Orbax-backed checkpointing with enough state for *exact* resumption.

Each checkpoint stores:

- ``state``: the full :class:`TrainState` pytree — parameters, optimizer
  state, step counter, and the root RNG key (saved as raw key data because
  extended PRNG dtypes are converted explicitly at the boundary).
- ``metadata``: JSON with the resolved config, tokenizer info, parameter
  count, library versions, and the best validation metric — sufficient to
  rebuild the model and detect config incompatibilities before restoring.

Saves go through ``orbax.checkpoint.CheckpointManager`` (atomic directory
renames, retention policies, async finalization). ``close()`` blocks until
pending async saves complete; the trainer calls it on every exit path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp

from jaxscale_lm import __version__
from jaxscale_lm.config import Config
from jaxscale_lm.training.state import TrainState
from jaxscale_lm.utils.logging import get_logger, log_event

_logger = get_logger("checkpoint")


def _to_savable(state: TrainState) -> dict[str, Any]:
    """TrainState -> plain dict, with the PRNG key lowered to uint32 data."""
    return {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
        "rng_key_data": jax.random.key_data(state.rng_key),
    }


def _from_savable(saved: dict[str, Any], template: TrainState) -> TrainState:
    """Rebuild a TrainState (re-wrapping the PRNG key) from restored data."""
    impl = jax.random.key_impl(template.rng_key)
    return TrainState(
        params=saved["params"],
        opt_state=saved["opt_state"],
        step=saved["step"],
        rng_key=jax.random.wrap_key_data(saved["rng_key_data"], impl=impl),
    )


def build_metadata(config: Config, num_params: int, best_metric_value: float | None) -> dict:
    """Checkpoint metadata: everything needed to rebuild and validate."""
    return {
        "format_version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "jaxscale_lm_version": __version__,
        "jax_version": jax.__version__,
        "config": config.model_dump(mode="json"),
        "model_config": config.model.model_dump(mode="json"),
        "tokenizer": config.tokenizer.model_dump(mode="json"),
        "parameter_count": num_params,
        "best_metric_name": config.checkpoint.best_metric,
        "best_metric_value": best_metric_value,
    }


@dataclass(frozen=True)
class CheckpointRef:
    """A resolved pointer into a checkpoint directory."""

    root: Path
    step: int | None  # None means "latest"


def resolve_checkpoint(path: str | Path) -> CheckpointRef:
    """Resolve CLI checkpoint paths.

    Accepts ``<run_dir>``, ``<run_dir>/latest``, or ``<run_dir>/<step>``.
    """
    p = Path(path)
    if p.name == "latest":
        return CheckpointRef(root=p.parent, step=None)
    if p.name.isdigit():
        return CheckpointRef(root=p.parent, step=int(p.name))
    return CheckpointRef(root=p, step=None)


class Checkpointer:
    """Thin lifecycle wrapper around ``ocp.CheckpointManager``."""

    def __init__(self, directory: Path, config: Config) -> None:
        self._config = config
        directory.mkdir(parents=True, exist_ok=True)
        opts = ocp.CheckpointManagerOptions(
            max_to_keep=config.checkpoint.max_to_keep,
            best_fn=(
                (lambda metrics: metrics[config.checkpoint.best_metric])
                if config.checkpoint.keep_best
                else None
            ),
            best_mode="min",
            create=True,
            enable_async_checkpointing=True,
        )
        self._manager = ocp.CheckpointManager(directory.resolve(), options=opts)
        self.directory = directory

    def save(
        self,
        state: TrainState,
        *,
        num_params: int,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        """Save at the state's current step. Returns whether a save started."""
        step = int(state.step)
        best_value = metrics.get(self._config.checkpoint.best_metric) if metrics else None
        saved = self._manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(_to_savable(state)),
                metadata=ocp.args.JsonSave(
                    build_metadata(self._config, num_params, best_value)
                ),
            ),
            metrics=metrics,
        )
        log_event(_logger, "checkpoint save", step=step, directory=str(self.directory))
        return bool(saved)

    def restore(self, template: TrainState, step: int | None = None) -> tuple[TrainState, dict]:
        """Restore ``step`` (or the latest) into the template's structure.

        Raises:
            FileNotFoundError: if no checkpoint exists.
            ValueError: if the stored model config conflicts with the current
                one (shapes would silently mismatch otherwise).
        """
        target = step if step is not None else self.latest_step()
        if target is None:
            raise FileNotFoundError(
                f"No checkpoint found under {self.directory}. "
                f"Run training first or check the --checkpoint path."
            )
        restored = self._manager.restore(
            target,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(_to_savable(template)),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        metadata = restored["metadata"]
        self._check_compatibility(metadata)
        state = _from_savable(restored["state"], template)
        log_event(_logger, "checkpoint restore", step=target, directory=str(self.directory))
        return state, metadata

    def _check_compatibility(self, metadata: dict) -> None:
        saved_model = metadata.get("model_config")
        current_model = self._config.model.model_dump(mode="json")
        if saved_model != current_model:
            diff = {
                k: (saved_model.get(k), current_model.get(k))
                for k in set(saved_model) | set(current_model)
                if saved_model.get(k) != current_model.get(k)
            }
            raise ValueError(
                f"Checkpoint model config is incompatible with the current config. "
                f"Differing fields (saved, current): {json.dumps(diff)}"
            )

    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    def all_steps(self) -> list[int]:
        return sorted(self._manager.all_steps())

    def wait(self) -> None:
        """Block until any in-flight async save is durable."""
        self._manager.wait_until_finished()

    def close(self) -> None:
        """Finalize async saves and release resources. Always call on exit."""
        self._manager.wait_until_finished()
        self._manager.close()


def read_metadata(directory: Path, step: int | None = None) -> tuple[int, dict]:
    """Read checkpoint metadata JSON (latest step if ``step`` is None).

    Returns ``(step, metadata)``. Used by inference/serving to rebuild the
    model config without a full Trainer.
    """
    manager = ocp.CheckpointManager(directory.resolve())
    try:
        target = step if step is not None else manager.latest_step()
        if target is None:
            raise FileNotFoundError(
                f"No checkpoint steps found under {directory}; train a model first."
            )
        restored = manager.restore(
            target, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())
        )
        result: dict = restored["metadata"]
        return target, result
    finally:
        manager.close()
