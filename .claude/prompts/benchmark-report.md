# Prompt: Benchmark Report Pass

Use this prompt when generating or validating benchmark output.

```text
You are improving the benchmark/reporting layer for JAXScale-LM.

Read:
- .claude/skills/benchmarking.md
- docs/benchmarking.md
- docs/results.md

Run only benchmarks supported by the current hardware. Do not fabricate unavailable measurements.

Ensure each benchmark records:
- raw samples
- git metadata
- dirty state
- package versions
- device info
- model config
- parameter count
- dtype
- warmup and measured iteration counts

Generate:
- JSONL raw records
- CSV summary
- Markdown summary
- PNG plots
- docs/results.md from actual outputs only

Clearly disclose hardware and limitations.
```

