# Benchmark evidence: run `20260703_175929_d8c6dc`

Committed copy of the raw evidence behind [docs/results.md](../../results.md).
Benchmark runs normally live under the gitignored `artifacts/benchmarks/`;
this one is preserved in git so every number in the results document can be
re-derived from tracked files. It supersedes run `20260611_085752_0eb57a`
(recorded on a dirty tree during pre-commit verification; retrievable from
git history), and was produced from a **clean working tree at the exact
commit recorded in every record**.

- **Run ID**: `20260703_175929_d8c6dc` (2026-07-03)
- **Command**: `UV_CACHE_DIR=.uv-cache uv run python scripts/benchmark.py
  --config configs/benchmark/default.yaml`
- **Hardware**: single Apple-Silicon CPU (arm64, macOS 26.5.1), JAX CPU
  backend, 1 device, 1 process. No GPU/TPU; no multi-device measurements.
- **Software**: Python 3.12.13, jax 0.10.1, jaxlib 0.10.1, flax 0.12.7,
  optax 0.2.8, orbax 0.12.0 (full inventory in each record's
  `environment` block)
- **Git commit**: `6bca6212cf2bf4f36d12c0f3e20c38f7bb1c9ff8`
  (**clean tree**: `git_dirty: false` in every record)

## Files

| file | contents |
|---|---|
| `records.jsonl` | 29 raw records (29 ok, 0 failed), one JSON object per line, `schema_version: 1`; each carries suite/name/mode, workload parameters, **raw samples in seconds**, derived stats (mean/median/std/p50/p90/p95/p99 where n suffices), status, and the environment block |
| `summary.csv` | flat per-record summary for spreadsheets |
| `summary.md` | human-readable per-suite tables (source of the results doc) |
| `resolved_config.yaml` | the fully resolved benchmark configuration |
| `plots/*.png` | the six documented plots rendered from these records |

## Schema notes

`schema_version` is 1 (`src/jaxscale_lm/benchmark/schema.py`). Timings are
seconds; tables in docs convert to milliseconds. Statistics can be
recomputed from `raw_samples_s` — nothing here is post-processed beyond
those derivations. p99 is withheld on every record because no measurement
uses ≥ 20 samples (the schema's honesty rule); percentiles beyond p50 are
likewise omitted for the 5-sample full-generation loops.
