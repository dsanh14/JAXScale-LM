# Benchmark Methodology

How every number in [results.md](results.md) is produced, and which
comparisons are valid.

## Core rules

1. **Synchronize before stopping any timer.** JAX dispatch is asynchronous;
   every timed region in
   [timing.py](../src/jaxscale_lm/utils/timing.py) ends with
   `jax.block_until_ready` on the outputs. A timer without a sync measures
   enqueue latency, not work.
2. **Separate compilation from steady state.** The *first* call of a jitted
   function includes tracing + XLA compilation and can be orders of
   magnitude slower than the second. We time the first call of a fresh jit
   explicitly (`mode: first_call`), then warm up, then measure
   `mode: steady_state`. The two are never mixed in one statistic.
3. **Warm up every compiled function** (`benchmark.warmup_iterations`,
   default 3) before steady-state measurement.
4. **Multiple repetitions, raw samples preserved.**
   `benchmark.measure_iterations` (default 10) samples per measurement;
   the JSONL records keep every raw sample, so statistics can be recomputed
   and outliers inspected. Reported: mean, median, std, p50/p90/p95, and
   p99 only when ≥ 20 samples exist.
5. **Failures are recorded, not dropped.** A failed suite or unsupported
   precision produces a `status: "failed"` record with the error text.
6. **Full disclosure in every record**: git commit + dirty flag, library
   versions, platform, device inventory, model config, dtype, batch size,
   prompt/sequence lengths, seed, iteration counts
   ([schema.py](../src/jaxscale_lm/benchmark/schema.py)).

## What each suite measures

| Suite | Measures | Notes |
|---|---|---|
| `compilation` | eager vs first jitted call vs warmed call; first-call cost per new shape | demonstrates tracing/XLA cost and shape-keyed caching |
| `training` | jitted train-step latency, tokens/s, examples/s across batch sizes and precisions; accumulated vs plain step | includes a best-effort memory snapshot per record |
| `prefill` | prompt-processing latency across batch sizes × prompt lengths | compute-bound phase |
| `decode` | single cached decode-step latency (ms/token) at different context lengths and batch sizes | measured on a fixed cache position, no sampling overhead |
| `e2e` | full generation: time-to-first-token, total latency, decode tokens/s | includes sampling + the per-step host sync of the EOS check |
| `cache` | naive full-prefix decoding vs KV-cached decoding, same workload | also asserts the two paths produce identical greedy output |

## Decode vs end-to-end numbers

`decode` isolates the compiled step function; `e2e` is what a client
experiences, including sampling, token bookkeeping, and one host-device
sync per generated token (the EOS check). On a CPU with a tiny model, that
sync overhead is a visible fraction of step time — which is honest, and
itself a lesson about asynchronous dispatch.

## Valid and invalid comparisons

Valid:
- two modes of the same workload in the same run (cached vs naive, eager vs
  jitted, batch 1 vs batch 4);
- the same suite across runs whose records show the same model config,
  dtype, platform, and library versions.

Invalid (the tooling labels but cannot stop you):
- comparing across different model configurations without saying so —
  records embed `model_config` precisely to make this checkable;
- presenting simulated-CPU-device numbers as multi-accelerator scaling
  (see [sharding.md](sharding.md));
- quoting bf16 "speedups" measured on CPU, where bf16 is emulated — the
  harness records the platform with every number;
- treating tiny-model results as representative of large-model behavior.
  Tiny workloads under-utilize devices and over-weight fixed overheads
  (dispatch, collectives); they demonstrate *mechanisms*, not production
  throughput.

## Hardware disclosure

Every record carries `environment.platform`, `device_names`,
`device_count`, and the host OS string. [results.md](results.md) opens with
the same disclosure. Numbers in this repository were measured on a CPU
unless a record says otherwise; no GPU/TPU results are claimed.

## Estimated FLOPs / MFU

Not reported. A defensible MFU number needs a documented FLOPs formula and
the hardware's peak-FLOPs assumption; for CPU runs of a toy model the
number would be noise dressed as rigor. The schema's `extra` field can
carry it if the project is run on accelerators with documented peaks.
