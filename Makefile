UV ?= uv
# Install the project non-editable: real files in site-packages, no .pth.
# On macOS, external processes (e.g. iCloud sync of ~/Documents) re-apply
# the UF_HIDDEN flag across .venv, and Python >= 3.12.4 silently skips
# hidden .pth files — which breaks *editable* imports at random times.
# Non-editable installs have no .pth, so they are immune. Source edits
# still propagate: [tool.uv] cache-keys includes src/**/*.py, and `uv run`
# syncs (rebuilding the wheel) whenever sources changed.
export UV_NO_EDITABLE := 1
RUN := $(UV) run

.PHONY: install venv-fix format lint typecheck test test-integration \
        test-all check-export verify smoke reproduce-cpu device-inspect \
        config-validate checkpoint-verify train-smoke evaluate-smoke \
        generate-smoke benchmark-smoke serve docker-build clean help

install: ## Sync the environment (incl. dev tools, non-editable project install)
	$(UV) sync --extra dev

venv-fix: ## (Darwin guard) heal a venv whose .pth files were hidden by an external editable sync
	@if [ "$$(uname)" = "Darwin" ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true; fi

format: venv-fix ## Auto-format with Ruff
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

lint: venv-fix ## Lint with Ruff (no changes)
	$(RUN) ruff format --check src tests scripts
	$(RUN) ruff check src tests scripts

typecheck: venv-fix ## Static type check with Pyright
	$(RUN) pyright

test: venv-fix ## Fast CPU unit tests
	$(RUN) pytest -m "not integration and not slow and not accelerator and not multi_device"

test-integration: venv-fix ## Integration tests (train/checkpoint/serve smoke)
	$(RUN) pytest -m "integration and not accelerator"

test-all: venv-fix ## Everything that runs on CPU
	$(RUN) pytest -m "not accelerator"

check-export: ## Verify the repo works from tracked files only (catches ignored sources)
	bash scripts/check_clean_export.sh

verify: ## Full acceptance gate: sync, import, format, lint, types, tests (fail-fast)
	$(UV) sync --extra dev
	$(RUN) python -c "import jaxscale_lm; print('import OK:', jaxscale_lm.__file__)"
	$(RUN) ruff format --check src tests scripts
	$(RUN) ruff check src tests scripts
	$(RUN) pyright
	$(RUN) pytest tests -q

device-inspect: ## Print the JAX device topology
	$(RUN) python scripts/inspect_devices.py

config-validate: ## Validate every shipped config (composition + cross-checks)
	$(RUN) python scripts/validate_configs.py

checkpoint-verify: ## Restore the smoke checkpoint and verify its contents
	$(RUN) python scripts/verify_checkpoint.py \
		--checkpoint artifacts/checkpoints/cpu_smoke/latest --restore

smoke: ## One-command CPU proof: devices → configs → train → restore → eval → generate → benchmark
	$(MAKE) device-inspect
	$(MAKE) config-validate
	@# Fresh training run: cpu_smoke checkpoints are scratch outputs of this
	@# target; removing them proves training (not resumption) on every run.
	rm -rf artifacts/checkpoints/cpu_smoke
	$(MAKE) train-smoke
	$(MAKE) checkpoint-verify
	$(MAKE) evaluate-smoke
	$(MAKE) generate-smoke
	$(MAKE) benchmark-smoke
	@echo "smoke: PASS (checkpoints + run manifest + benchmark artifacts under artifacts/)"

reproduce-cpu: ## Full CPU reproduction from a fresh clone: verify + smoke
	$(MAKE) verify
	$(MAKE) smoke
	@echo "reproduce-cpu: PASS"

train-smoke: venv-fix ## 10-step CPU training run on synthetic data
	$(RUN) python scripts/train.py --config configs/train/cpu_smoke.yaml

evaluate-smoke: venv-fix ## Evaluate the smoke checkpoint
	$(RUN) python scripts/evaluate.py --checkpoint artifacts/checkpoints/cpu_smoke/latest

generate-smoke: venv-fix ## Greedy generation from the smoke checkpoint
	$(RUN) python scripts/generate.py --checkpoint artifacts/checkpoints/cpu_smoke/latest \
		--prompt "Once upon a time" --max-new-tokens 32 --use-kv-cache

benchmark-smoke: venv-fix ## Quick benchmark sweep (CPU)
	$(RUN) python scripts/benchmark.py --config configs/benchmark/default.yaml --quick

serve: venv-fix ## Serve the smoke checkpoint locally
	$(RUN) python scripts/serve.py --checkpoint artifacts/checkpoints/cpu_smoke/latest \
		--host 127.0.0.1 --port 8000

docker-build: ## Build the CPU serving image
	docker build -t jaxscale-lm:latest .

clean: ## Remove caches and build outputs (keeps artifacts/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help: ## Show targets
	@grep -E '^[a-zA-Z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "%-20s %s\n", $$1, $$2}'
