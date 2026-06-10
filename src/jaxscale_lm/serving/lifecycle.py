"""Model lifecycle management for serving.

One :class:`ModelManager` owns at most one loaded engine. Loading goes
REGISTERED -> LOADING -> (warmup) -> READY, and readiness is only reported
after the compiled inference path is warm — a cold model would otherwise
serve its first request with multi-second compile latency.

Concurrency model: generation requests are **serialized** with a lock. JAX
inference here is single-model, single-device, and CPU-bound; concurrent
tracing/execution would interleave on the same device without any speedup.
No dynamic batching is implemented (documented in docs/limitations.md).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from jaxscale_lm.config import InferenceConfig, ServingConfig
from jaxscale_lm.inference.engine import GenerationResult, InferenceEngine
from jaxscale_lm.serving import metrics
from jaxscale_lm.serving.registry import ModelEntry, ModelRegistry, ModelStatus
from jaxscale_lm.training.checkpoint import resolve_checkpoint
from jaxscale_lm.utils.logging import get_logger, log_event

_logger = get_logger("serving")


class ModelManager:
    """Owns the active engine and its registry entry."""

    def __init__(self, registry: ModelRegistry, serving_config: ServingConfig) -> None:
        self.registry = registry
        self.serving_config = serving_config
        self._engine: InferenceEngine | None = None
        self._active_model_id: str | None = None
        self._lock = threading.Lock()  # serializes generation (see module doc)

    # -- state ----------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def active_model_id(self) -> str | None:
        return self._active_model_id

    @property
    def engine(self) -> InferenceEngine:
        if self._engine is None:
            raise RuntimeError(
                "No model is loaded. POST /v1/models/load with a checkpoint path first."
            )
        return self._engine

    # -- lifecycle -------------------------------------------------------------
    def load(self, checkpoint_path: str, model_id: str | None = None) -> tuple[str, float, float]:
        """Load a checkpoint; returns (model_id, load_seconds, warmup_seconds).

        Registers the model if unknown, then LOADING -> READY with warmup in
        between. On failure the entry is marked FAILED and the error re-raised.
        """
        ref = resolve_checkpoint(checkpoint_path)
        resolved_id = model_id or ref.root.name
        log_event(_logger, "model load requested", model_id=resolved_id, path=checkpoint_path)

        start = time.perf_counter()
        try:
            engine = InferenceEngine.from_checkpoint(checkpoint_path)
        except Exception:
            metrics.MODEL_LOAD_FAILURES.inc()
            self._mark_failed_if_known(resolved_id, "checkpoint load failed")
            raise
        load_s = time.perf_counter() - start
        metrics.MODEL_LOAD_SECONDS.observe(load_s)

        if resolved_id not in {e.model_id for e in self.registry.list()}:
            self.registry.register(
                ModelEntry(
                    model_id=resolved_id,
                    version=1,
                    checkpoint_path=str(Path(checkpoint_path)),
                    training_step=engine.checkpoint_step,
                    parameter_count=_parameter_count(engine),
                    max_sequence_length=engine.model_config.max_sequence_length,
                    precision=engine.model_config.compute_dtype,
                    model_config=engine.model_config.model_dump(mode="json"),
                )
            )
        self.registry.set_status(resolved_id, ModelStatus.LOADING)

        warmup_s = 0.0
        try:
            if self.serving_config.warmup:
                warmup_s = engine.warmup()
                metrics.WARMUP_SECONDS.set(warmup_s)
        except Exception:
            self.registry.set_status(resolved_id, ModelStatus.FAILED, "warmup failed")
            raise

        with self._lock:
            self._engine = engine
            previous = self._active_model_id
            self._active_model_id = resolved_id
        if previous and previous != resolved_id:
            self.registry.set_status(previous, ModelStatus.UNLOADED, "replaced")
        self.registry.set_status(resolved_id, ModelStatus.READY)
        log_event(
            _logger,
            "model ready",
            model_id=resolved_id,
            load_seconds=round(load_s, 3),
            warmup_seconds=round(warmup_s, 3),
        )
        return resolved_id, load_s, warmup_s

    def unload(self, model_id: str) -> None:
        """Unload the active model (safe: requests are serialized by the lock)."""
        with self._lock:
            if model_id != self._active_model_id:
                raise KeyError(
                    f"Model {model_id!r} is not the active model "
                    f"({self._active_model_id!r}); nothing to unload."
                )
            self._engine = None
            self._active_model_id = None
        self.registry.set_status(model_id, ModelStatus.UNLOADED)
        log_event(_logger, "model unloaded", model_id=model_id)

    def _mark_failed_if_known(self, model_id: str, detail: str) -> None:
        try:
            entry = self.registry.get(model_id)
        except KeyError:
            return
        if ModelStatus(entry.status) in (ModelStatus.LOADING,):
            self.registry.set_status(model_id, ModelStatus.FAILED, detail)

    # -- inference ---------------------------------------------------------
    def generate(self, prompt: str, options: InferenceConfig) -> GenerationResult:
        """Serialized generation through the active engine."""
        engine = self.engine  # raises RuntimeError when nothing is loaded
        with self._lock:
            return engine.generate(prompt, options)


def _parameter_count(engine: InferenceEngine) -> int:
    from jaxscale_lm.utils.tree import count_params

    return count_params(engine._params)  # noqa: SLF001 - manager is a friend class
