# Benchmark evidence: run `20260611_085752_0eb57a`

Committed copy of the raw evidence behind [docs/results.md](../../results.md).
Benchmark runs normally live under the gitignored `artifacts/benchmarks/`;
this one is preserved in git so every number in the results document can be
re-derived from tracked files.

- **Run ID**: `20260611_085752_0eb57a` (2026-06-11)
- **Command**: `uv run python scripts/benchmark.py --config configs/benchmark/default.yaml`
  (as invoked at the time; the Makefile now standardizes on `uv run --no-sync`)
- **Hardware**: single Apple-Silicon CPU (arm64, macOS 26.5.1), JAX CPU
  backend, 1 device, 1 process. No GPU/TPU; no multi-device measurements.
- **Software**: Python 3.12.13, jax 0.10.1, flax 0.12.7, optax 0.2.8,
  orbax 0.12.0 (full inventory in each record's `environment` block)
- **Git commit**: `7dfa39040182ebc09189774757adc79b025163a7` (dirty tree —
  run was made while verifying that revision pre-commit)

## Files

| file | contents |
|---|---|
| `records.jsonl` | 29 raw records, one JSON object per line, `schema_version: 1`; each carries suite/name/mode, workload parameters, **raw samples in seconds**, derived stats (mean/median/std/p50/p90/p95/p99 where n suffices), status, and the environment block |
| `summary.csv` | flat per-record summary for spreadsheets |
| `summary.md` | human-readable per-suite tables (source of the results doc) |
| `resolved_config.yaml` | the fully resolved benchmark configuration |
| `plots/*.png` | the six documented plots rendered from these records |

## Schema notes

`schema_version` is 1 (`src/jaxscale_lm/benchmark/schema.py`). Timings are
seconds; tables in docs convert to milliseconds. Statistics can be
recomputed from `raw_samples_s` — nothing here is post-processed beyond
those derivations.
