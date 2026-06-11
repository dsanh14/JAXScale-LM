# JAXScale-LM — Implementation Plan

Status: living document. Updated as milestones complete.

## 1. Current repository state (Milestone 0 audit)

Audited 2026-06-10.

- The repository is **empty**: a fresh `git init` on branch `main` with zero
  commits. The only file present is a stray `.DS_Store` (will be gitignored).
- There is no existing code, configuration, or documentation to reuse or
  replace. Everything below is greenfield.
- **Host environment**:
  - macOS (Darwin 25.5.0), Apple Silicon (aarch64), 12 CPU cores.
  - No CUDA GPU and no TPU. JAX will run on the **CPU backend** with a single
    device by default.
  - System Python is 3.9.6 (too old). `uv` 0.11.x has been installed and will
    manage a project-local Python **3.12** toolchain and the lock file.
- **Implication for multi-device work**: real accelerator scaling cannot be
  measured here. Multi-device *logic* (mesh construction, named sharding,
  data-parallel placement) will be implemented correctly and tested using
  XLA's host-platform device-count override
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`, set **before** JAX
  initialization). All documentation and benchmark records will clearly label
  these as simulated CPU devices, never as accelerator scaling.

## 2. Proposed architecture

A single installable package `jaxscale_lm` under `src/`, with thin CLI
scripts in `scripts/` and YAML configs in `configs/`. Layout follows the
structure in the project brief (configs/, src/jaxscale_lm/{utils,data,model,
training,distributed,inference,serving,benchmark}, scripts/, tests/, docs/,
dashboards/, artifacts/).

### Key technical decisions

| Decision | Choice | Rationale |
|---|---|---|
| Python | 3.12 (managed by uv) | Newest version with mature wheel coverage for jax/flax/orbax; ≥3.11 as required. |
| NN library | **Flax NNX** | Current recommended Flax API; explicit state, plays well with `jax.jit` via `nnx.jit`/functional split. |
| Positional encoding | **RoPE** | Preferred by spec; modest complexity; pairs naturally with KV-cache decode (position-indexed rotation). |
| Tokenizer | Byte-level (vocab 256 + specials) as default; Hugging Face `tokenizers` BPE as the trainable path | Byte tokenizer is fully deterministic and dependency-light for tests/smoke; BPE for the "real" small model. Both behind one `Tokenizer` protocol. |
| Dataset | TinyStories via `datasets`, plus local plain-text fallback and a deterministic in-memory synthetic fixture for tests | Spec preference; synthetic fixture keeps CI hermetic/offline. |
| Sequence construction | Concatenate-and-chunk into fixed-length blocks | Primary path per spec; no padding needed in training, simpler loss masking. |
| Training step | Pure function over `(params-as-state, optimizer state, batch, rng)` jitted with `jax.jit`; gradient accumulation via `jax.lax.scan` over microbatches | Keeps the jitted function pure; scan keeps compile size constant in accumulation steps. |
| Mixed precision | Separate `param_dtype` / `compute_dtype`; loss & metric reductions in float32 | Explicit dtype boundaries; bf16 is safe on CPU and the relevant mode for accelerators. |
| Checkpointing | `orbax.checkpoint.CheckpointManager` with a composite of model state, optimizer state, RNG key, step, and JSON metadata (resolved config, tokenizer info, best metric) | Faithful resumption requires all of these; manager handles retention/atomicity. |
| Sharding | `jax.sharding.Mesh` with axes `("data", "model")`; `NamedSharding` + `PartitionSpec`. Data parallelism shards the batch dim over `data`; params replicated. Optional model-axis sharding of large matmuls is documented but not the default. | Current unified `jax.Array` model; no `pmap`. |
| KV cache | Fixed-capacity arrays `[batch, max_seq_len, num_kv_heads, head_dim]` updated with `jax.lax.dynamic_update_slice_in_dim` at the current position | Fixed shapes → one compile for all decode steps; indexed update avoids re-concatenation. |
| Serving | FastAPI + lifespan-managed engine, Pydantic v2 schemas, `prometheus_client`; requests **serialized** through a lock (documented — no dynamic batching claimed) | Honest concurrency story; warmup before readiness. |
| Registry | JSON file with atomic `os.replace` writes | Lightweight; inspired by managed-model lifecycles, explicitly not a cloud clone. |
| Benchmarks | Versioned dataclass records → JSONL raw + CSV/Markdown summaries + Matplotlib PNGs; `block_until_ready` around every timed region; warmup separated from steady state | Spec §17. |
| Config | Pydantic v2 models loaded from YAML, validated before any model init; resolved config saved with every run | Single typed config system shared by all CLIs. |

### Deviations from the suggested layout

Recorded as implementation progressed:

- `model/config.py` was **not** created: `ModelConfig` lives in the central
  `config.py` so cross-section validation (e.g. tokenizer vocab ==
  model vocab, data sequence length <= model context) happens in one place.
- `model/embeddings.py` holds both the token embedding and the RoPE
  helpers; RoPE is consumed inside attention and did not warrant a file.
- `training/state.py` contains `TrainState` as a NamedTuple pytree (params,
  optimizer state, step, root RNG key) rather than a Flax TrainState class —
  smaller surface, identical capability, checkpoint-friendly.
- Dropout RNGs are passed *at call time* (`rngs=` argument through the
  module tree) instead of being stored in modules, keeping the model state
  pure parameters and the jitted train step explicitly seeded.
- `benchmark/inference.py` covers the prefill/decode/e2e/cache suites in
  one module (they share a jitted-function setup); the suite names in
  records still match the spec's categories.

## 3. Milestones

- **M0** Repository audit + this plan. ✅
- **M1** Foundation: pyproject + uv.lock, package skeleton, typed config
  loader, structured logging, seeding utils, timing utils, device
  inspection script, Ruff + Pyright + Pytest wiring, Makefile, .gitignore. ✅
- **M2** Data: tokenizer protocol (byte + BPE), download script, packing
  (concat-and-chunk), deterministic loader, synthetic fixture, tests. ✅
- **M3** Model: embeddings, RoPE, causal attention (train/prefill/decode
  paths), MLP, block, transformer, KV-cache structure; shape + causality +
  numerical tests; parameter-count utility. ✅
- **M4** Training: loss/metrics, AdamW + schedule + weight-decay masking,
  gradient clipping, gradient accumulation (scan), jitted train step,
  evaluation loop, trainer, CPU smoke training run. ✅
- **M5** Checkpointing: Orbax manager wrapper, metadata, save/restore,
  exact-resumption integration test (N+M vs interrupted N→restore→M). ✅
- **M6** Inference: sampling (greedy/temperature/top-k/top-p), naive
  generation, prefill+decode with KV cache, equivalence and determinism
  tests, generate CLI. ✅
- **M7** Distributed: mesh builder, partitioning specs, placement helpers,
  diagnostics, single-device fallback, simulated multi-device CPU tests. ✅
- **M8** Benchmarks: schema, compilation/training/prefill/decode/e2e/cache
  comparison suites, memory probe, runner, plots, benchmark CLI. ✅
- **M9** Serving: FastAPI app, schemas, registry, lifecycle/warmup,
  Prometheus metrics, integration tests via httpx/ASGI. ✅
- **M10** Packaging & docs: Dockerfile, docker-compose, README, all docs/,
  Mermaid diagrams, run real CPU benchmarks → docs/results.md. ✅

### Final verification pass (2026-06-11)

Fixes applied while bringing the full gate green:

- Lint/format: 21 files reformatted, 17 Ruff findings fixed
  (`StrEnum` for `ModelStatus`, explicit `zip(strict=)`, unused locals).
- Pyright: 16 errors fixed — `loss_stats` accepts host NumPy arrays,
  `optax.apply_updates` result cast, Optional coalescing in plots,
  jaxlib version via `importlib.metadata`, metadata None-guard in
  `Checkpointer._check_compatibility`, monitoring-listener signature.
- `optax.global_norm` → `optax.tree.norm` (deprecation).
- **Resumption semantics**: `Trainer.train(until_step=)` added so the
  exact-resumption test interrupts a run *without* shrinking
  `max_steps` — a shorter horizon changes the cosine-decay schedule and
  the first N updates would legitimately differ.
- **Checkpoint restore order**: metadata is restored and validated
  *before* the state so incompatible configs fail with a clear
  ValueError instead of an Orbax shape error; `read_metadata` raises
  FileNotFoundError for missing directories (serving maps it to 404).
- macOS env quirk: uv-installed venv files carry `UF_HIDDEN`, which
  Python ≥ 3.12.4 skips for `.pth` files; `make install` now clears the
  flag and `RUN` pins `--extra dev` to avoid sync churn (README note).
- Real benchmark run `20260611_085752_0eb57a` (29 records, 0 failed)
  transcribed into docs/results.md.
- Verified end-to-end on this machine: make train-smoke /
  evaluate-smoke / generate-smoke / benchmark-smoke, `--resume latest`,
  all script `--help`s, live serving (health/ready/models/generate/
  metrics), verify_checkpoint, download_data, inspect_devices.

## 4. Expected risks

1. **API drift** (Flax NNX / Orbax move quickly). Mitigation: resolve current
   versions with uv first, code against the installed APIs, lock with
   `uv.lock`.
2. **No accelerator** — all performance numbers are CPU numbers. Mitigation:
   hardware disclosure in every benchmark record and in docs/limitations.md;
   simulated-device runs clearly labeled.
3. **bf16 on CPU is slow** (emulated). Precision *correctness* is testable on
   CPU; precision *speedups* are not. Documented, not fabricated.
4. **Orbax async saves** can race with process exit. Mitigation: explicit
   `wait_until_finished()` in trainer shutdown paths.
5. **Recompilation hazards** in benchmarks (shape changes, unstable static
   args). Mitigation: compile-counter instrumentation + fixed-shape decode.
6. **TinyStories download** requires network. Mitigation: local-text and
   synthetic paths keep every test and smoke workflow offline.

## 5. Testing strategy

- Pytest with markers: `unit`, `integration`, `slow`, `accelerator`,
  `multi_device`. Default CI selection runs on CPU only.
- Unit: config validation, tokenizer round-trips, packing, masks, shapes,
  loss, schedule, sampling filters, cache updates, registry transitions,
  benchmark stats.
- Numerical: causality (future-token perturbation does not change past
  logits), cached-vs-full logit closeness, grad-accumulation vs large-batch
  closeness, token-weighted eval aggregation, greedy determinism.
- Integration: tiny train → save → restore → exact resume; evaluate;
  generate; FastAPI app lifecycle + endpoints via ASGI transport.
- Regression: fixed-seed loss trajectory tolerance, parameter counts for the
  preset configs, benchmark record schema, deterministic generation fixture.
- Multi-device tests run in a subprocess with
  `XLA_FLAGS=--xla_force_host_platform_device_count=8` so the main test
  process keeps its default backend.

## 6. Benchmarking strategy

- Every timed region: warmup iterations → `block_until_ready` → N measured
  repetitions, each ended with `block_until_ready`.
- First-call (compile+execute) timed separately from steady state.
- Records carry schema_version, git commit + dirty flag, library versions,
  platform, device inventory, full workload parameters, raw samples, and
  derived stats (mean/median/std/p50/p90/p95/p99 where n suffices).
- Outputs under a configurable `artifacts/benchmarks/<run_id>/` directory:
  `records.jsonl`, `summary.csv`, `summary.md`, `plots/*.png`.
- Failed runs recorded with `status: "failed"` and the error, never dropped.

## 7. Definition of done

All acceptance criteria from the project brief (§27), specifically:

- Tiny model trains on CPU without NaNs; params change after a step.
- Causal masking proven by test; cached decode ≈ full decode within
  documented tolerance; grad accumulation ≈ large batch within tolerance.
- Checkpoint resume reproduces uninterrupted training within tolerance.
- Eval metrics token-weighted; perplexity from aggregated NLL.
- Jitted train step; compile vs steady-state measured with device sync.
- Data-parallel path works under simulated devices; single-device fallback.
- Serving: health/ready/load/generate/metrics endpoints pass integration
  tests; warmup precedes readiness.
- `make format lint typecheck test` pass; integration smoke passes.
- README commands verified by actually running them.
- docs/results.md contains only numbers produced by real runs on this
  machine, with hardware disclosure.
