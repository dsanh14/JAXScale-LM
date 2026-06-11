"""Compilation benchmarks: eager vs first jitted call vs steady state, and
shape-driven recompilation.

Methodology notes:
- "first_call" is timed on a *fresh* jitted function (compile + execute).
- "steady_state" warms up first; samples measure execution only.
- "eager" runs the un-jitted function (op-by-op dispatch).
- A new sequence length re-times the first call of a fresh jit to expose
  recompilation cost; the jit cache counter confirms the recompile happened.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from jaxscale_lm.config import Config
from jaxscale_lm.benchmark.schema import BenchmarkRecord, record_from_timing
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.utils.timing import compilation_count, measure, time_synchronized


def _forward_fn(graphdef: nnx.GraphDef) -> Callable[[nnx.State, jax.Array], jax.Array]:
    def forward(params: nnx.State, ids: jax.Array) -> jax.Array:
        model = nnx.merge(graphdef, params)
        logits, _ = model(ids, deterministic=True)
        return logits

    return forward


def run(config: Config) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    bench = config.benchmark
    model = build_model(config.model, config.project.seed)
    graphdef, params = nnx.split(model)
    forward = _forward_fn(graphdef)
    batch = bench.batch_sizes[0]

    for seq_len in bench.sequence_lengths:
        ids = jax.random.randint(
            jax.random.key(bench.seed), (batch, seq_len), 0, config.model.vocab_size
        )
        common = {
            "batch_size": batch,
            "sequence_length": seq_len,
            "seed": bench.seed,
            "dtype": config.model.compute_dtype,
        }

        # Eager (no jit): op-by-op dispatch. Loop variables are bound via
        # default args so each closure is self-contained.
        eager = measure(
            lambda ids=ids: forward(params, ids),
            warmup=min(bench.warmup_iterations, 1),
            iterations=max(bench.measure_iterations // 2, 3),
        )
        records.append(
            record_from_timing("compilation", f"forward_seq{seq_len}", "eager", eager, **common)
        )

        # First call of a fresh jit: compile + execute.
        jitted = jax.jit(forward)
        first_s = time_synchronized(lambda jitted=jitted, ids=ids: jitted(params, ids))
        records.append(
            BenchmarkRecord(
                suite="compilation",
                name=f"forward_seq{seq_len}",
                mode="first_call",
                raw_samples_s=[first_s],
                mean_s=first_s,
                median_s=first_s,
                measure_iterations=1,
                warmup_iterations=0,
                **common,
            )
        )

        # Steady state of the same jit.
        steady = measure(
            lambda jitted=jitted, ids=ids: jitted(params, ids),
            warmup=bench.warmup_iterations,
            iterations=bench.measure_iterations,
        )
        rec = record_from_timing(
            "compilation", f"forward_seq{seq_len}", "steady_state", steady, **common
        )
        rec.extra["first_call_over_steady"] = first_s / max(steady.median_s, 1e-12)
        rec.extra["process_compilations_so_far"] = compilation_count()
        records.append(rec)

    # Shape recompilation: one jit, three shape changes; first-call cost
    # recurs for each new (batch, seq) signature.
    jitted = jax.jit(forward)
    shapes = [
        (batch, bench.sequence_lengths[0]),
        (batch, bench.sequence_lengths[-1]),  # seq change -> recompile
        (batch * 2, bench.sequence_lengths[0]),  # batch change -> recompile
    ]
    for b, s in shapes:
        ids = jax.random.randint(jax.random.key(bench.seed), (b, s), 0, config.model.vocab_size)
        before = compilation_count()
        elapsed = time_synchronized(lambda ids=ids: jitted(params, ids))
        compiled = compilation_count() - before
        records.append(
            BenchmarkRecord(
                suite="compilation",
                name="shape_recompilation",
                mode="first_call_per_shape",
                raw_samples_s=[elapsed],
                mean_s=elapsed,
                median_s=elapsed,
                batch_size=b,
                sequence_length=s,
                seed=bench.seed,
                dtype=config.model.compute_dtype,
                measure_iterations=1,
                warmup_iterations=0,
                extra={"recompiled": bool(compiled)},
            )
        )
    return records
