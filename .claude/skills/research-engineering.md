# Skill: Senior Research Engineering

Use this skill when improving the project’s credibility, architecture, tests, documentation, or research-style presentation.

## Goal

Make JAXScale-LM read like the work of a careful ML systems/research engineer: measurable, auditable, reproducible, and technically honest.

## What Good Looks Like

- The README makes clear claims and links each claim to a command, test, doc, or artifact.
- The docs explain tradeoffs instead of pretending everything is production-grade.
- The code favors small, typed, testable functions over clever abstractions.
- Numerical invariants are tested, not merely described.
- Benchmark methodology is explicit enough that another engineer can reproduce the run.

## High-Value Improvements

Prefer these before adding optional features:

1. Fix import, packaging, and test reliability.
2. Add one-command CPU reproduction.
3. Add run manifests with environment, git, config, and device metadata.
4. Strengthen numerical equivalence tests.
5. Generate `docs/results.md` only from real benchmark artifacts.
6. Improve architecture diagrams and limitation docs.

## Anti-Patterns

- Adding a feature just because it sounds impressive.
- Claiming model quality from a tiny smoke run.
- Reporting benchmark averages without raw samples.
- Calling simulated CPU devices multi-accelerator scaling.
- Leaving TODOs in core paths.
- Swallowing exceptions and returning “best effort” success.

## Output Style

After each milestone, report:

- Files changed
- Design decisions
- Commands executed
- Tests passed
- Tests failed
- Known limitations
- Next milestone

