# Skill: Benchmarking

Use this skill when adding or changing benchmark code, benchmark docs, or result reporting.

## Benchmark Contract

Every benchmark record should include:

- schema version
- run id
- timestamp
- git commit and dirty state
- Python/JAX/jaxlib/Flax/Optax/Orbax versions
- platform and device info
- process count
- model config and parameter count
- dtype
- batch size
- prompt length
- generated token count
- seed
- warmup iterations
- measured iterations
- raw samples
- mean, median, stddev, p50, p90, p95, p99 when valid

## Timing Rules

- Warm up compiled functions.
- Separate first-call compile-plus-execute from steady-state execution.
- Call `block_until_ready()` before stopping timers.
- Preserve failed runs with failure metadata.
- Never compare materially different configs without labeling them.

## Output Files

Benchmark runs should write under a configurable directory, usually:

```text
artifacts/benchmarks/<run_id>/
  records.jsonl
  summary.csv
  summary.md
  plots/
```

`docs/results.md` must be derived from real benchmark output. If no valid benchmark run exists, say that explicitly.

## Senior-Level Commentary

Benchmark docs should explain:

- asynchronous dispatch
- XLA compilation cost
- shape recompilation
- first-token latency vs decode throughput
- why tiny workloads can underutilize hardware
- why CPU measurements do not imply accelerator scaling

