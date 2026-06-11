"""Lightweight local model registry.

Tracks model versions and their lifecycle states in a single JSON file with
atomic writes (write-temp + ``os.replace``). The design is *inspired by*
managed-model lifecycle patterns (register -> load -> ready -> unload), but
it is deliberately a small local tool, not a cloud-service clone.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ModelStatus(StrEnum):
    REGISTERED = "REGISTERED"
    LOADING = "LOADING"
    READY = "READY"
    FAILED = "FAILED"
    UNLOADED = "UNLOADED"


# Legal lifecycle transitions; anything else is a bug worth surfacing.
_TRANSITIONS: dict[ModelStatus, set[ModelStatus]] = {
    ModelStatus.REGISTERED: {ModelStatus.LOADING},
    ModelStatus.LOADING: {ModelStatus.READY, ModelStatus.FAILED},
    # READY -> LOADING covers reload-on-restart of an already-known model.
    ModelStatus.READY: {ModelStatus.UNLOADED, ModelStatus.FAILED, ModelStatus.LOADING},
    ModelStatus.FAILED: {ModelStatus.LOADING},
    ModelStatus.UNLOADED: {ModelStatus.LOADING},
}


@dataclass
class ModelEntry:
    """One registered model version."""

    model_id: str
    version: int
    checkpoint_path: str
    training_step: int
    parameter_count: int
    max_sequence_length: int
    precision: str
    status: str = ModelStatus.REGISTERED.value
    validation_loss: float | None = None
    tokenizer_path: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    status_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """JSON-file-backed registry with atomic persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, ModelEntry] = {}
        if path.exists():
            self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        with self.path.open() as f:
            raw = json.load(f)
        if raw.get("registry_version") != 1:
            raise ValueError(
                f"Unsupported registry version in {self.path}: "
                f"{raw.get('registry_version')!r} (expected 1)."
            )
        self._entries = {key: ModelEntry(**value) for key, value in raw["models"].items()}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registry_version": 1,
            "models": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        # Atomic write: temp file in the same directory, then rename.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # -- API -----------------------------------------------------------------
    def register(self, entry: ModelEntry) -> ModelEntry:
        if entry.model_id in self._entries:
            raise ValueError(
                f"Model id {entry.model_id!r} is already registered; "
                f"unload/re-register under a new version."
            )
        self._entries[entry.model_id] = entry
        self._persist()
        return entry

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            raise KeyError(f"Unknown model id {model_id!r}; registered: {sorted(self._entries)}")
        return self._entries[model_id]

    def list(self) -> list[ModelEntry]:
        return sorted(self._entries.values(), key=lambda e: e.created_at)

    def set_status(
        self, model_id: str, status: ModelStatus, detail: str | None = None
    ) -> ModelEntry:
        entry = self.get(model_id)
        current = ModelStatus(entry.status)
        if status not in _TRANSITIONS[current]:
            raise ValueError(
                f"Illegal status transition {current.value} -> {status.value} "
                f"for model {model_id!r}."
            )
        entry.status = status.value
        entry.status_detail = detail
        self._persist()
        return entry
