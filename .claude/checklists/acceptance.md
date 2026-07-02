# Acceptance Checklist

Use this checklist before calling the project polished.

## Reliability

- [ ] `UV_CACHE_DIR=.uv-cache uv sync --extra dev` succeeds.
- [ ] `UV_CACHE_DIR=.uv-cache uv run python -c "import jaxscale_lm; print(jaxscale_lm.__file__)"` succeeds.
- [ ] `UV_CACHE_DIR=.uv-cache uv run ruff format --check src tests scripts` succeeds.
- [ ] `UV_CACHE_DIR=.uv-cache uv run ruff check src tests scripts` succeeds.
- [ ] `UV_CACHE_DIR=.uv-cache uv run pyright` succeeds.
- [ ] `UV_CACHE_DIR=.uv-cache uv run pytest tests -q` succeeds.

## CPU Smoke Workflow

- [ ] Device inspection works.
- [ ] Tiny training works without NaNs.
- [ ] Checkpoint save works.
- [ ] Checkpoint restore works.
- [ ] Evaluation works.
- [ ] Generation works.
- [ ] Benchmark smoke writes JSONL, CSV, Markdown, and plots.
- [ ] Serving health, ready, generate, and metrics endpoints work in tests.

## Numerical Correctness

- [ ] Causal masking prevents future-token leakage.
- [ ] KV-cached decode matches full-prefix decode within tolerance.
- [ ] Gradient accumulation matches equivalent large-batch update within tolerance.
- [ ] Checkpoint resume matches uninterrupted training within tolerance.
- [ ] Evaluation metrics are token-weighted.
- [ ] Greedy generation is deterministic.

## Documentation Honesty

- [ ] Hardware disclosure is present.
- [ ] Limitations are explicit.
- [ ] `docs/results.md` contains only measured results or says no results are available.
- [ ] Multi-device results are not claimed unless measured on real devices.
- [ ] Simulated CPU devices are labeled as simulation.

