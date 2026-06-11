"""Training-step benchmarks: latency and throughput over batch sizes and
precisions, plus gradient-accumulation throughput comparison."""

from __future__ import annotations

import jax
import numpy as np

from jaxscale_lm.benchmark.memory import snapshot
from jaxscale_lm.benchmark.schema import BenchmarkRecord, record_from_timing
from jaxscale_lm.config import Config
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.training.optimizer import build_optimizer
from jaxscale_lm.training.state import create_train_state
from jaxscale_lm.training.step import make_train_step
from jaxscale_lm.types import Batch
from jaxscale_lm.utils.device import supports_dtype
from jaxscale_lm.utils.seed import make_key
from jaxscale_lm.utils.timing import measure


def _synthetic_batch(accum: int, micro: int, seq: int, vocab: int, seed: int) -> Batch:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=(accum, micro, seq + 1)).astype(np.int32)
    return Batch(
        input_ids=jax.numpy.asarray(ids[..., :-1]),
        target_ids=jax.numpy.asarray(ids[..., 1:]),
        loss_mask=jax.numpy.ones((accum, micro, seq), jax.numpy.float32),
    )


def _measure_step(
    config: Config, batch_size: int, accum: int, dtype: str, seq_len: int
) -> tuple[BenchmarkRecord, int]:
    bench = config.benchmark
    model_cfg = config.model.model_copy(update={"compute_dtype": dtype})
    model = build_model(model_cfg, config.project.seed)
    n_params = model.num_params()
    tx, schedule = build_optimizer(config.optimizer, config.training.max_steps)
    graphdef, state = create_train_state(model, tx, make_key(config.project.seed))
    step_fn = jax.jit(make_train_step(graphdef, tx, schedule, accum))
    batch = _synthetic_batch(accum, batch_size, seq_len, model_cfg.vocab_size, bench.seed)

    # The state must be threaded through, so time a closure with rebind.
    holder = {"state": state}

    def run_once():
        holder["state"], metrics = step_fn(holder["state"], batch)
        return metrics["loss"]

    timing = measure(run_once, warmup=bench.warmup_iterations, iterations=bench.measure_iterations)
    tokens_per_step = accum * batch_size * seq_len
    record = record_from_timing(
        "training",
        f"train_step_b{batch_size}x{accum}_{dtype}",
        "steady_state",
        timing,
        batch_size=batch_size,
        sequence_length=seq_len,
        dtype=dtype,
        seed=bench.seed,
        parameter_count=n_params,
    )
    record.extra.update(
        {
            "accumulation_steps": accum,
            "tokens_per_second": tokens_per_step / timing.median_s,
            "examples_per_second": accum * batch_size / timing.median_s,
            "final_loss": float(jax.device_get(run_once())),
            "memory": snapshot().to_dict(),
        }
    )
    return record, n_params


def run(config: Config) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    bench = config.benchmark
    seq_len = config.data.sequence_length

    for dtype in bench.precisions:
        if not supports_dtype(dtype):
            records.append(
                BenchmarkRecord(
                    suite="training",
                    name=f"train_step_{dtype}",
                    mode="steady_state",
                    status="failed",
                    error=(
                        f"compute dtype {dtype} unsupported on backend '{jax.default_backend()}'"
                    ),
                    dtype=dtype,
                )
            )
            continue
        for batch_size in bench.batch_sizes:
            record, _ = _measure_step(config, batch_size, 1, dtype, seq_len)
            records.append(record)

    # Gradient accumulation: same effective batch, micro vs accumulated.
    biggest = max(bench.batch_sizes)
    if biggest >= 2:
        accum_record, _ = _measure_step(
            config, biggest // 2, 2, config.model.compute_dtype, seq_len
        )
        accum_record.name = f"train_step_accum2_b{biggest // 2}"
        records.append(accum_record)
    return records
