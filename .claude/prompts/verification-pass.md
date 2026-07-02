# Prompt: Verification Pass

Use this prompt when the implementation is mostly present and needs hardening.

```text
You are performing a verification and hardening pass on JAXScale-LM.

Do not add new features unless required to make existing claims true.

Run or fix the following:

UV_CACHE_DIR=.uv-cache uv sync --extra dev
UV_CACHE_DIR=.uv-cache uv run python -c "import jaxscale_lm; print(jaxscale_lm.__file__)"
UV_CACHE_DIR=.uv-cache uv run ruff format --check src tests scripts
UV_CACHE_DIR=.uv-cache uv run ruff check src tests scripts
UV_CACHE_DIR=.uv-cache uv run pyright
UV_CACHE_DIR=.uv-cache uv run pytest tests -q

Then run the smallest CPU smoke workflow:
- inspect devices
- train tiny config
- verify checkpoint restore
- evaluate
- generate
- benchmark smoke

For every failure, identify whether it is:
- code defect
- packaging defect
- test defect
- environment limitation

Fix code/package/test defects. Document environment limitations.
Do not claim success for commands that did not pass.
```

