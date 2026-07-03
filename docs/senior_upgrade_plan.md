# Senior Upgrade Plan

Status: living document. Audit date: 2026-07-03. Supersedes nothing —
complements `docs/implementation_plan.md` (the greenfield build plan) with a
hardening/verification view of the finished system.

## 1. Current repository state

- Branch `main`, clean tree at audit start (last commit `a155666`).
- Installable package `jaxscale_lm` under `src/`, thin CLIs in `scripts/`,
  YAML configs in `configs/`, tests split unit/integration/regression
  (148 test functions, 154 collected items), docs in `docs/`, CI in
  `.github/workflows/ci.yml` (Linux CPU: lint, typecheck, full CPU tests,
  clean-export check).
- All six acceptance commands pass on this host (see §9) after the
  packaging fix described in §5.
- Committed evidence: one full benchmark run under
  `docs/benchmarks/20260611_085752_0eb57a/` (records JSONL + summaries +
  plots), referenced by `docs/results.md`; local (gitignored) artifacts
  under `artifacts/benchmarks/` and `artifacts/checkpoints/cpu_smoke/`.

## 2. Current strengths

- **Numerical invariants are tested, not described**: gradient-accumulation
  ≈ large-batch equivalence, interrupted-vs-uninterrupted checkpoint resume,
  KV-cached vs naive greedy equality, future-token/causal-mask isolation,
  token-weighted (not batch-averaged) eval metrics, deterministic greedy
  decode, dropout-only-in-training.
- **Benchmark records are self-describing** (`benchmark/schema.py`,
  `schema_version = 1`): git commit + dirty flag, Python/JAX/jaxlib/Flax/
  Optax/Orbax versions, platform/device/process info, model config,
  parameter count, dtype, batch/prompt/generate lengths, seed, warmup and
  measured iteration counts, raw samples, mean/median/std/p50/p90/p95 and
  p99 only when n ≥ 20. Failures are recorded with metadata, not dropped.
- **`docs/results.md` is provenance-first**: every number traces to a
  committed run directory; hardware disclosure and "CPU ≠ accelerator"
  caveats are explicit; multi-device is labeled simulation-only.
- **Honest docs**: limitations, sharding-as-correctness-not-scaling, JAX
  concepts, benchmarking methodology, reproducibility notes all exist.
- **Serving is tested**: health/ready/generate/metrics endpoints, warmup
  gating readiness, registry lifecycle, error paths with actionable 4xx.
- Clean-export CI check guards against gitignore rules silently dropping
  tracked sources (a real past incident).

## 3. Current weaknesses

- **(Fixed in this pass)** Editable-install imports broke at random times
  on this host: an external macOS process (iCloud sync of `~/Documents`)
  recursively re-applies `UF_HIDDEN` across `.venv`, and Python ≥ 3.12.4
  silently skips hidden `.pth` files. Details and the fix in §5.
- **No run manifest for training runs.** Training persists checkpoints and
  `resolved_config.yaml` only. There is no
  `artifacts/runs/<run_id>/{environment.json, git.json, metrics.jsonl}`;
  step/eval metrics exist only as log lines. Benchmarks embed environment
  data per record, but training runs do not capture it at all.
- **No single verify/reproduce entrypoint.** The smoke workflow is five
  separate make targets; nothing chains device inspection → config
  validation → train → restore → eval → generate → benchmark and fails
  loudly as one command.
- Flax NNX deprecation warnings (`.value` access in `model/embeddings.py`)
  clutter test output; will become errors on a future Flax upgrade.
- `docs/implementation_plan.md` still reads as the build-time plan; fine as
  history, but the repo lacked this hardening-status document.

## 4. Highest-risk gaps (ranked)

1. **Import reliability** — was intermittent, environment-triggered, and
   made every downstream command fail. Fixed first (§5).
2. **Training-run reproducibility manifest** — without environment/git/
   metrics capture, a training run cannot be audited after the fact; this
   is the largest remaining gap versus `.claude/skills/reproducibility.md`.
3. **One-command reproduction** — reviewers will not run five targets in
   order; a broken step could go unnoticed.
4. **Warning debt** — NNX `.value` deprecations are a future breakage.

## 5. Packaging / import status

**Root cause found and fixed.** The whole `.venv` tree gets `UF_HIDDEN`
re-applied asynchronously (observed twice within minutes during this
audit) by a macOS process external to the repo — `~/Documents` is
iCloud-managed on this host. Python ≥ 3.12.4 skips hidden `.pth` files, so
the *editable* install's `jaxscale_lm.pth` stopped reaching `sys.path` at
unpredictable times, while non-`.pth` imports are unaffected by the flag.

Fix (three mutually reinforcing layers, no behavior change elsewhere):

- Makefile exports `UV_NO_EDITABLE=1`: the project installs **non-editable**
  (real files in site-packages, no `.pth` to skip — verified immune even
  with `chflags -R hidden` applied to the installed package).
  `[tool.uv] cache-keys` includes `src/**/*.py`, and Makefile targets drop
  `--no-sync`, so `uv run` rebuilds the wheel whenever sources change
  (verified with a probe edit).
- `[tool.uv] reinstall-package = ["jaxscale-lm"]`: every sync — including
  the implicit sync in a raw `uv run` — reinstalls the project, so even an
  editable `uv sync` starts Python with a freshly written, non-hidden
  `.pth` (the external flag-setter needs minutes; the window is now
  microseconds).
- `tests/conftest.py` raises an actionable error naming the failure mode
  and the remedy if the import still fails.

Non-editable trade-off accepted: tracebacks point into site-packages
rather than `src/`; in exchange, imports no longer depend on `.pth`
processing at all. `make venv-fix` remains as a healer for externally
created editable installs.

## 6. Testing status

All 154 tests pass (`uv run pytest tests -q`, exit 0). Mapping against the
required invariant list — every item has at least one existing test:

| Required invariant | Covering test(s) |
|---|---|
| package import | exercised by every test via `conftest.py`; clean-export check imports submodules explicitly |
| config validation | `test_config.py` (defaults merge, unknown keys, divisibility, dropout/vocab/schedule bounds, actionable missing-file errors) |
| tokenizer encode/decode | `test_round_trip_ascii`, `test_round_trip_unicode`, `test_specials_skipped_on_decode`, BPE `test_train_and_round_trip` |
| data packing | `test_shapes_and_shift`, `test_eos_separates_documents`, short-text pad/drop cases |
| shifted targets | `test_shapes_and_shift` |
| causal masking | `test_future_tokens_do_not_affect_past_logits`, `test_mask_excludes_tokens` |
| model forward shapes | `test_attention_shape`, `test_mlp_shape`, `test_transformer_logits_shape_and_dtype`, `test_init_shapes` |
| finite loss | `test_loss_finite_and_params_change`, `test_uniform_logits_log_vocab` |
| optimizer update changes params | `test_loss_finite_and_params_change`, `test_loss_decreases_on_repeated_batch` |
| grad accumulation ≈ large batch | `test_accumulated_matches_large_batch` |
| checkpoint resume ≡ continued training | `test_interrupted_equals_uninterrupted`, `test_restore_specific_step`, `test_start_step_fast_forward_matches` |
| KV-cached ≡ full-prefix decode | `test_cached_decode_matches_full_forward`, `test_cached_equals_naive_greedy` |
| deterministic greedy generation | `test_greedy_cached_deterministic`, `test_greedy_deterministic_via_api`, `test_greedy_fixture_stable` |
| token-weighted eval metrics | `test_token_weighted_not_batch_averaged` |
| benchmark schema stability | `test_schema_fields_stable`, `test_record_round_trips_to_dict`, `test_required_keys_present` |
| FastAPI health/ready/metrics | `test_health`, `test_ready_after_startup_warmup`, `test_metrics_exposed_and_updated` |

Remaining test work: none blocking; keep the suite warning-clean (§4.4).

## 7. Benchmark / reporting status

Compliant with `.claude/skills/benchmarking.md`:

- Records carry every required field (verified against a real
  `records.jsonl`); raw samples always preserved; p99 withheld under 20
  samples rather than fabricated; failed runs recorded with `status` +
  `error`.
- Output layout: `artifacts/benchmarks/<run_id>/{records.jsonl,
  summary.csv, summary.md, plots/, resolved_config.yaml}`.
- `docs/results.md` cites only the committed run
  `docs/benchmarks/20260611_085752_0eb57a/` with git commit, versions,
  hardware disclosure, and explicit "not measured" statements for anything
  unmeasured. No fabricated numbers found.

Remaining benchmark work: none blocking. Optional: regenerate results on
the current commit once this pass lands (old run is labeled `git_dirty`).

## 8. Documentation status

Present and accurate: README (purpose, why-JAX, architecture Mermaid,
install, smoke, API, methodology, hardware disclosure, limitations,
roadmap), `docs/architecture.md`, `docs/jax_concepts.md`,
`docs/sharding.md`, `docs/benchmarking.md`, `docs/reproducibility.md`,
`docs/limitations.md`, `docs/results.md`.

Needs updating in this pass: README install section (the editable-install
macOS caveat is superseded by the non-editable model), reproducibility doc
(run-manifest layout once implemented), README/docs mention of the new
`make verify` / `make smoke` / `make reproduce-cpu` entrypoints.

## 9. Exact commands that must pass

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
UV_CACHE_DIR=.uv-cache uv run python -c "import jaxscale_lm; print(jaxscale_lm.__file__)"
UV_CACHE_DIR=.uv-cache uv run ruff format --check src tests scripts
UV_CACHE_DIR=.uv-cache uv run ruff check src tests scripts
UV_CACHE_DIR=.uv-cache uv run pyright
UV_CACHE_DIR=.uv-cache uv run pytest tests -q
```

Status 2026-07-03: **all six pass** on this host (post-fix), plus:

```bash
make verify          # the six commands above, chained, fail-fast
make smoke           # device → config → train → restore → eval → generate → benchmark
make reproduce-cpu   # smoke + run manifest under artifacts/runs/<run_id>/
```

All three make targets are implemented and verified passing on this host
(`make reproduce-cpu` exit 0, 2026-07-03).

## 10. Prioritized roadmap

1. **DONE — packaging/import reliability** (§5). All acceptance commands
   green.
2. **DONE — make targets**: `verify` (the six commands, fail-fast),
   `smoke` (device inspection → validation of every shipped config →
   *fresh* tiny train with NaN guard → checkpoint restore verification →
   eval → generate → benchmark smoke writing JSONL/CSV/Markdown/plots),
   `reproduce-cpu` (= verify + smoke). All three verified passing.
3. **DONE — run manifest**: `environment_info()` factored into
   `utils/environment.py` (benchmark schema unchanged, additive
   `git_branch` only); `utils/run_manifest.py` writes
   `artifacts/runs/<run_id>/{resolved_config.yaml, environment.json,
   git.json, run.json, metrics.jsonl, checkpoints -> symlink}` on every
   trainer invocation. Covered by
   `tests/integration/test_run_manifest.py`. Exact-resumption semantics
   unchanged (checkpoints stay at their stable path).
4. **DONE — warning hygiene**: NNX `.value` reads migrated to the
   `variable[...]` API; test output no longer emits Flax deprecation
   warnings.
5. **DONE — documentation polish**: README install/reproduction sections
   rewritten for the non-editable model and `make reproduce-cpu`;
   `docs/reproducibility.md` documents the run-manifest layout.
6. Remaining, optional (explicitly *after* everything above, per
   `.claude/skills/research-engineering.md`): fresh full benchmark run on
   a clean commit to refresh `docs/results.md`; roadmap features (GQA
   runtime path, prompt bucketing, etc.) remain out of scope for this
   pass.

## 11. Definition of done

- The six §9 commands pass from a fresh `git clone` with only `uv`
  installed (CI proves the Linux path; this host proves macOS).
- `make verify`, `make smoke`, and `make reproduce-cpu` each succeed as a
  single command; `reproduce-cpu` leaves a run directory containing
  resolved config, environment, git state, and step/eval metrics.
- Every checklist item in `.claude/checklists/acceptance.md` is checked or
  has a documented, honest exception.
- No claim in README/docs exceeds what a committed artifact, test, or
  command demonstrates; simulated CPU devices remain labeled simulation.
- Working tree committed with no generated artifacts added to git beyond
  intentional evidence fixtures under `docs/benchmarks/`.
