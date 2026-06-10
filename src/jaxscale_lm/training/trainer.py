"""The training orchestrator.

Responsibilities (host-side; never inside jit):
- build tokenizer, data, model, optimizer, jitted step functions
- run the train loop with periodic eval / checkpoint / logging
- detect unexpected recompilation via a jit-cache counter
- save the resolved config next to the checkpoints
- guarantee async checkpoint finalization on every exit path
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jaxscale_lm.config import Config, save_resolved_config
from jaxscale_lm.data.loader import DataBundle, build_data, eval_batches, train_batches
from jaxscale_lm.data.tokenizer import build_tokenizer
from jaxscale_lm.distributed.diagnostics import describe_mesh
from jaxscale_lm.distributed.mesh import build_mesh
from jaxscale_lm.distributed.partitioning import (
    eval_batch_sharding,
    replicated,
    train_batch_sharding,
    validate_batch_divisibility,
)
from jaxscale_lm.distributed.placement import place_batch, place_tree
from jaxscale_lm.model.transformer import Transformer, build_model
from jaxscale_lm.training.checkpoint import Checkpointer
from jaxscale_lm.training.metrics import MetricAggregator
from jaxscale_lm.training.optimizer import build_optimizer
from jaxscale_lm.training.state import TrainState, create_train_state
from jaxscale_lm.training.step import make_eval_step, make_train_step
from jaxscale_lm.types import Batch
from jaxscale_lm.utils.logging import get_logger, log_event
from jaxscale_lm.utils.seed import make_key
from jaxscale_lm.utils.timing import jit_cache_size

_logger = get_logger("trainer")


class Trainer:
    """Single-host trainer with mesh-based data parallelism.

    Batches are sharded over the mesh's ``data`` axis and parameters are
    replicated; on a single device the same code path degenerates to plain
    replicated execution with zero collectives.
    """

    def __init__(self, config: Config, *, data: DataBundle | None = None) -> None:
        self.config = config
        self.tokenizer = build_tokenizer(config.tokenizer)
        self.data = data or build_data(config.data, self.tokenizer, config.project.seed)

        # Mesh: identical code path for 1 or N devices. Batches shard over
        # the 'data' axis; parameters/optimizer state are replicated.
        self.mesh = build_mesh(config.distributed)
        validate_batch_divisibility(config.data.batch_size, self.mesh, config.distributed)
        self._replicated = replicated(self.mesh)
        self._batch_sharding = train_batch_sharding(self.mesh, config.distributed)
        self._eval_sharding = eval_batch_sharding(self.mesh)

        self.model: Transformer = build_model(config.model, config.project.seed)
        self.num_params = self.model.num_params()

        tx, schedule = build_optimizer(config.optimizer, config.training.max_steps)
        self.graphdef, self.state = create_train_state(
            self.model, tx, make_key(config.project.seed)
        )
        self.state = place_tree(self.state, self._replicated)
        self._train_step = jax.jit(
            make_train_step(self.graphdef, tx, schedule, config.training.gradient_accumulation_steps)
        )
        self._eval_step = jax.jit(make_eval_step(self.graphdef))

        accum = config.training.gradient_accumulation_steps
        data_devices = self.mesh.shape[config.distributed.axis_names[0]]
        effective_batch = config.data.batch_size * accum
        log_event(
            _logger,
            "trainer initialized",
            parameters=self.num_params,
            microbatch_per_device=config.data.batch_size // data_devices,
            microbatch_size=config.data.batch_size,
            accumulation_steps=accum,
            data_parallel_devices=data_devices,
            process_count=jax.process_count(),
            effective_global_batch=effective_batch,
            effective_tokens_per_step=effective_batch * config.data.sequence_length,
            platform=jax.default_backend(),
        )
        for line in describe_mesh(self.mesh):
            _logger.debug(line)

        self.checkpointer = Checkpointer(config.checkpoint_dir, config)
        save_resolved_config(config, config.checkpoint_dir / "resolved_config.yaml")

    # -- data ---------------------------------------------------------------
    def _stacked_train_batches(self, start_step: int) -> Iterator[Batch]:
        """Yield batches shaped [accum, micro, seq] for the jitted step."""
        accum = self.config.training.gradient_accumulation_steps
        micro = self.config.data.batch_size
        for batch in train_batches(
            self.data.train,
            batch_size=micro * accum,
            seed=self.config.project.seed,
            shuffle=self.config.data.shuffle,
            start_step=start_step,
        ):
            yield Batch(
                *(np.asarray(x).reshape(accum, micro, *np.asarray(x).shape[1:]) for x in batch)
            )

    # -- checkpoint ---------------------------------------------------------
    def resume(self, step: int | None = None) -> int:
        """Restore the latest (or a specific) checkpoint into the trainer."""
        self.state, metadata = self.checkpointer.restore(self.state, step)
        self.state = place_tree(self.state, self._replicated)
        restored_step = int(self.state.step)
        log_event(
            _logger,
            "resumed from checkpoint",
            step=restored_step,
            saved_at=metadata.get("created_at"),
        )
        return restored_step

    # -- evaluation ----------------------------------------------------------
    def evaluate(self) -> dict[str, float]:
        """Token-weighted evaluation over the validation split."""
        aggregator = MetricAggregator()
        for batch in eval_batches(
            self.data.validation,
            batch_size=self.config.data.batch_size,
            num_batches=self.config.evaluation.num_batches,
        ):
            stats = self._eval_step(
                self.state.params, place_batch(batch, self._eval_sharding)
            )
            aggregator.update(jax.device_get(stats))
        return aggregator.summary()

    # -- training -----------------------------------------------------------
    def train(self) -> dict[str, float]:
        """Run to ``training.max_steps``; returns the final eval summary."""
        cfg = self.config
        start_step = int(self.state.step)
        if start_step >= cfg.training.max_steps:
            log_event(_logger, "nothing to do", step=start_step, max_steps=cfg.training.max_steps)
            try:
                return self.evaluate()
            finally:
                self.checkpointer.close()

        batches = self._stacked_train_batches(start_step)
        last_eval: dict[str, float] = {}
        compile_count_seen = 0
        window_start = time.perf_counter()
        window_tokens = 0.0

        try:
            for step in range(start_step, cfg.training.max_steps):
                batch = place_batch(next(batches), self._batch_sharding)
                self.state, metrics = self._train_step(self.state, batch)

                if (step + 1) % cfg.training.log_interval == 0:
                    metrics = jax.device_get(metrics)
                    loss = float(metrics["loss"])
                    if not np.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss {loss} at step {step + 1}; "
                            f"lower optimizer.learning_rate or check the data pipeline."
                        )
                    window_tokens += float(metrics["valid_tokens"]) * cfg.training.log_interval
                    elapsed = time.perf_counter() - window_start
                    log_event(
                        _logger,
                        "train step",
                        step=step + 1,
                        loss=round(loss, 4),
                        accuracy=round(float(metrics["accuracy"]), 4),
                        grad_norm=round(float(metrics["grad_norm"]), 4),
                        lr=f"{float(metrics['learning_rate']):.2e}",
                        tokens_per_s=round(window_tokens / max(elapsed, 1e-9)),
                    )
                    window_start = time.perf_counter()
                    window_tokens = 0.0

                    cache_size = jit_cache_size(self._train_step)
                    if cache_size is not None and cache_size > max(compile_count_seen, 1):
                        _logger.warning(
                            "train step recompiled",
                            extra={"compilations": cache_size, "step": step + 1},
                        )
                    compile_count_seen = max(compile_count_seen, cache_size or 0)

                if (step + 1) % cfg.evaluation.interval_steps == 0:
                    last_eval = self.evaluate()
                    log_event(_logger, "evaluation", step=step + 1, **last_eval)

                if (step + 1) % cfg.checkpoint.interval_steps == 0:
                    self.checkpointer.save(
                        self.state,
                        num_params=self.num_params,
                        metrics={"validation_loss": last_eval.get("loss", float("inf"))},
                    )

            # Final eval + save if the last step wasn't on a boundary.
            if cfg.training.max_steps % cfg.evaluation.interval_steps != 0 or not last_eval:
                last_eval = self.evaluate()
                log_event(_logger, "final evaluation", **last_eval)
            if cfg.training.max_steps % cfg.checkpoint.interval_steps != 0:
                self.checkpointer.save(
                    self.state,
                    num_params=self.num_params,
                    metrics={"validation_loss": last_eval.get("loss", float("inf"))},
                )
            return last_eval
        finally:
            # Async saves must be durable before the process exits, on both
            # success and failure paths.
            self.checkpointer.close()
