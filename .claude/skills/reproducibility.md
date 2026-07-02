# Skill: Reproducibility

Use this skill when touching configs, training runs, checkpoints, datasets, tokenizers, or artifact layout.

## Reproducibility Contract

A reproducible run should capture:

- resolved config
- git commit
- dirty working-tree status
- Python and package versions
- JAX backend and devices
- random seed
- dataset source/version or synthetic data parameters
- tokenizer metadata
- model config and parameter count
- checkpoint step and metadata
- benchmark raw samples when benchmarking

## Recommended Run Layout

```text
artifacts/runs/<run_id>/
  resolved_config.yaml
  environment.json
  git.json
  metrics.jsonl
  checkpoints/
  benchmarks/
  plots/
```

Generated artifacts should be excluded from Git unless they are tiny intentional fixtures.

## Checkpoint Correctness

Checkpoint restore is credible only if it includes:

- model parameters/state
- optimizer state
- global step
- RNG state
- relevant metadata
- model configuration
- tokenizer metadata or reference
- best validation metric when tracked

Prefer exact-resumption tests over “loads without crashing” tests.

