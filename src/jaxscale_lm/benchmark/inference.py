"""Inference benchmarks: prefill, decode, end-to-end generation, and the
naive-vs-cached comparison.

Decode latency is measured at steady state on a *fixed* cache position
(functional cache updates mean re-running the same step is idempotent), so
the number is a clean per-token cost without loop or sampling overhead.
End-to-end numbers include sampling and the per-step host sync and are
reported separately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jaxscale_lm.benchmark.schema import BenchmarkRecord, record_from_timing
from jaxscale_lm.config import Config
from jaxscale_lm.inference.decode import make_cached_decode_fn, make_naive_decode_fn
from jaxscale_lm.inference.generate import cached_generate, naive_generate
from jaxscale_lm.inference.prefill import make_prefill_fn
from jaxscale_lm.inference.sampling import SamplingParams
from jaxscale_lm.model.cache import init_cache
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.utils.seed import make_key
from jaxscale_lm.utils.timing import measure


class _Setup:
    def __init__(self, config: Config) -> None:
        model = build_model(config.model, config.project.seed)
        self.parameter_count = model.num_params()
        self.graphdef, self.params = nnx.split(model)
        self.prefill = jax.jit(make_prefill_fn(self.graphdef))
        self.decode = jax.jit(make_cached_decode_fn(self.graphdef))
        self.naive = jax.jit(make_naive_decode_fn(self.graphdef))
        self.vocab = config.model.vocab_size
        self.capacity = config.model.max_sequence_length

    def prompt(self, batch: int, length: int, seed: int) -> jax.Array:
        return jax.random.randint(jax.random.key(seed), (batch, length), 0, self.vocab)


def run_prefill(config: Config, setup: _Setup) -> list[BenchmarkRecord]:
    records = []
    bench = config.benchmark
    for batch in bench.batch_sizes:
        for prompt_len in bench.prompt_lengths:
            if prompt_len >= setup.capacity:
                continue
            ids = setup.prompt(batch, prompt_len, bench.seed)
            cache = init_cache(config.model, batch, setup.capacity)
            timing = measure(
                lambda ids=ids, cache=cache: setup.prefill(setup.params, ids, cache)[0],
                warmup=bench.warmup_iterations,
                iterations=bench.measure_iterations,
            )
            rec = record_from_timing(
                "prefill",
                f"prefill_b{batch}_p{prompt_len}",
                "steady_state",
                timing,
                batch_size=batch,
                prompt_length=prompt_len,
                dtype=config.model.compute_dtype,
                seed=bench.seed,
                parameter_count=setup.parameter_count,
            )
            rec.extra["prompt_tokens_per_second"] = batch * prompt_len / timing.median_s
            records.append(rec)
    return records


def run_decode(config: Config, setup: _Setup) -> list[BenchmarkRecord]:
    records = []
    bench = config.benchmark
    for batch in bench.batch_sizes:
        for prompt_len in bench.prompt_lengths:
            if prompt_len + 1 >= setup.capacity:
                continue
            ids = setup.prompt(batch, prompt_len, bench.seed)
            cache = init_cache(config.model, batch, setup.capacity)
            logits, cache = setup.prefill(setup.params, ids, cache)
            token = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
            timing = measure(
                lambda token=token, cache=cache: setup.decode(setup.params, token, cache)[0],
                warmup=bench.warmup_iterations,
                iterations=bench.measure_iterations,
            )
            rec = record_from_timing(
                "decode",
                f"decode_b{batch}_ctx{prompt_len}",
                "steady_state",
                timing,
                batch_size=batch,
                prompt_length=prompt_len,
                dtype=config.model.compute_dtype,
                seed=bench.seed,
                parameter_count=setup.parameter_count,
            )
            rec.extra["ms_per_token"] = timing.median_s * 1000
            rec.extra["tokens_per_second"] = batch / timing.median_s
            rec.extra["cache_capacity"] = setup.capacity
            records.append(rec)
    return records


def run_e2e(config: Config, setup: _Setup) -> list[BenchmarkRecord]:
    records = []
    bench = config.benchmark
    sampling = SamplingParams()  # greedy: deterministic across repetitions
    for prompt_len in bench.prompt_lengths:
        for gen_len in bench.generate_lengths:
            if prompt_len + gen_len > setup.capacity:
                continue
            ids = setup.prompt(1, prompt_len, bench.seed)

            def generate_once():
                cache = init_cache(config.model, 1, setup.capacity)
                return cached_generate(
                    setup.prefill,
                    setup.decode,
                    setup.params,
                    ids,
                    cache,
                    max_new_tokens=gen_len,
                    sampling=sampling,
                    key=make_key(bench.seed),
                    eos_id=None,  # fixed step count for stable statistics
                    pad_id=0,
                    vocab_size=setup.vocab,
                )

            # Warmup compiles prefill+decode for this shape.
            for _ in range(max(bench.warmup_iterations, 1)):
                generate_once()
            outputs = [generate_once() for _ in range(max(bench.measure_iterations // 2, 3))]

            total = [o.timing.total_s for o in outputs]
            ttft = [o.timing.prefill_s for o in outputs]
            decode_s = [o.timing.decode_s for o in outputs]
            n = len(outputs)
            rec = BenchmarkRecord(
                suite="e2e",
                name=f"generate_p{prompt_len}_g{gen_len}",
                mode="steady_state",
                raw_samples_s=total,
                mean_s=sum(total) / n,
                median_s=sorted(total)[n // 2],
                batch_size=1,
                prompt_length=prompt_len,
                generate_length=gen_len,
                dtype=config.model.compute_dtype,
                seed=bench.seed,
                parameter_count=setup.parameter_count,
                warmup_iterations=max(bench.warmup_iterations, 1),
                measure_iterations=n,
                extra={
                    "time_to_first_token_s_median": sorted(ttft)[n // 2],
                    "decode_s_median": sorted(decode_s)[n // 2],
                    "generated_tokens_per_second": gen_len / (sorted(decode_s)[n // 2] or 1e-9),
                },
            )
            records.append(rec)
    return records


def run_cache_comparison(config: Config, setup: _Setup) -> list[BenchmarkRecord]:
    """Naive full-prefix decoding vs KV-cached decoding, same workload."""
    records = []
    bench = config.benchmark
    sampling = SamplingParams()
    prompt_len = bench.prompt_lengths[0]
    for gen_len in bench.generate_lengths:
        if prompt_len + gen_len > setup.capacity:
            continue
        ids = setup.prompt(1, prompt_len, bench.seed)

        def cached_once():
            cache = init_cache(config.model, 1, setup.capacity)
            return cached_generate(
                setup.prefill,
                setup.decode,
                setup.params,
                ids,
                cache,
                max_new_tokens=gen_len,
                sampling=sampling,
                key=make_key(bench.seed),
                eos_id=None,
                pad_id=0,
                vocab_size=setup.vocab,
            )

        def naive_once():
            return naive_generate(
                setup.naive,
                setup.params,
                ids,
                capacity=setup.capacity,
                max_new_tokens=gen_len,
                sampling=sampling,
                key=make_key(bench.seed),
                eos_id=None,
                pad_id=0,
                vocab_size=setup.vocab,
            )

        for name, fn in (("cached", cached_once), ("naive", naive_once)):
            for _ in range(max(bench.warmup_iterations, 1)):
                out = fn()
            outs = [fn() for _ in range(max(bench.measure_iterations // 2, 3))]
            decode_samples = [o.timing.decode_s for o in outs]
            n = len(outs)
            rec = BenchmarkRecord(
                suite="cache",
                name=f"{name}_p{prompt_len}_g{gen_len}",
                mode="steady_state",
                raw_samples_s=decode_samples,
                mean_s=sum(decode_samples) / n,
                median_s=sorted(decode_samples)[n // 2],
                batch_size=1,
                prompt_length=prompt_len,
                generate_length=gen_len,
                dtype=config.model.compute_dtype,
                seed=bench.seed,
                parameter_count=setup.parameter_count,
                warmup_iterations=max(bench.warmup_iterations, 1),
                measure_iterations=n,
                extra={
                    "kv_cache": name == "cached",
                    "ms_per_token": sorted(decode_samples)[n // 2] * 1000 / gen_len,
                    "buffer_capacity": setup.capacity,
                },
            )
            records.append(rec)
        # Equivalence sanity check (greedy outputs must match).
        if not (cached_once().token_ids == naive_once().token_ids).all():
            records.append(
                BenchmarkRecord(
                    suite="cache",
                    name=f"equivalence_p{prompt_len}_g{gen_len}",
                    mode="check",
                    status="failed",
                    error="cached and naive greedy outputs diverged",
                )
            )
    return records


def run(config: Config, suites: set[str]) -> list[BenchmarkRecord]:
    setup = _Setup(config)
    records: list[BenchmarkRecord] = []
    if "prefill" in suites:
        records += run_prefill(config, setup)
    if "decode" in suites:
        records += run_decode(config, setup)
    if "e2e" in suites:
        records += run_e2e(config, setup)
    if "cache" in suites:
        records += run_cache_comparison(config, setup)
    return records
