# CLAUDE.md

Guidance for Claude Code agents working on JAXScale-LM.

## Project Identity

JAXScale-LM is a portfolio-grade ML systems lab, not a model-quality demo. The project should demonstrate rigorous engineering around JAX/XLA training, inference, checkpointing, sharding, benchmarking, serving, and reproducibility.

Do not claim production scale, Google/DeepMind affiliation, accelerator scaling, memory numbers, benchmark results, or model quality unless those claims are measured and documented in this repository.

## First Principles

1. Reliability before feature breadth.
2. Correctness before optimization.
3. Measured results before claims.
4. Reproducibility before polish.
5. Small, focused changes before large rewrites.

## Required First Actions

Before implementing nontrivial changes:

1. Inspect the repository state.
2. Read `docs/implementation_plan.md`.
3. Check `git status --short`.
4. Identify whether generated artifacts or user changes are present.
5. Avoid reverting or deleting work you did not create.

## Core Commands

Use the local uv cache when sandboxed environments cannot access the default uv cache:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
UV_CACHE_DIR=.uv-cache uv run python -c "import jaxscale_lm; print(jaxscale_lm.__file__)"
UV_CACHE_DIR=.uv-cache uv run ruff format --check src tests scripts
UV_CACHE_DIR=.uv-cache uv run ruff check src tests scripts
UV_CACHE_DIR=.uv-cache uv run pyright
UV_CACHE_DIR=.uv-cache uv run pytest tests -q
```

If one of these fails, fix that before adding new capability.

## Engineering Rules

- Keep jitted functions pure: no file I/O, logging side effects, hidden randomness, or Python mutation that affects semantics.
- Randomness must be explicit through JAX PRNG keys.
- Host-side data loading must stay outside jitted functions.
- Timing must call `block_until_ready()` before stopping timers.
- Benchmark records must preserve raw samples and hardware/software metadata.
- Evaluation loss and perplexity must be token-weighted, not batch-averaged.
- Checkpoints must include enough state for faithful resumption.
- Multi-device code must clearly distinguish real accelerator/device scaling from simulated CPU devices.
- Error messages should include context and an actionable next step.

## Definition of Done

A change is done only when:

1. The relevant tests pass.
2. Lint and type checks pass or exceptions are documented.
3. Documentation reflects the behavior.
4. Generated outputs are written under configurable artifact directories.
5. Claims in docs or README are backed by actual code, tests, or benchmark artifacts.

## Useful Local Guidance

Read the focused files under `.claude/` as needed:

- `.claude/skills/research-engineering.md`
- `.claude/skills/jax-systems.md`
- `.claude/skills/benchmarking.md`
- `.claude/skills/reproducibility.md`
- `.claude/checklists/acceptance.md`
- `.claude/prompts/senior-upgrade.md`
- `.claude/prompts/verification-pass.md`

