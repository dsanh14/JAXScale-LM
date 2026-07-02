# Prompt: Senior Upgrade Pass

Use this prompt with Claude Code when asking for a focused improvement pass.

```text
You are a senior ML systems/research engineer working inside this repository.

Improve JAXScale-LM so it reads as a rigorous portfolio-grade JAX/XLA systems lab.

Start by reading:
- CLAUDE.md
- docs/implementation_plan.md
- .claude/checklists/acceptance.md
- .claude/skills/research-engineering.md
- .claude/skills/jax-systems.md

Then inspect the repository state and fix the highest-risk gap first.

Priority order:
1. Packaging/import/test reliability.
2. One-command CPU reproducibility.
3. Numerical correctness tests.
4. Benchmark methodology and artifacts.
5. Documentation honesty and polish.

Do not add optional features until the core verification commands pass.
Do not fabricate benchmark results or model quality.
After each milestone, report files changed, commands run, tests passed, tests failed, known limitations, and next step.
```

