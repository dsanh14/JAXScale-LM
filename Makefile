UV ?= uv
RUN := $(UV) run

.PHONY: install format lint typecheck test test-integration test-all \
        check-export train-smoke evaluate-smoke generate-smoke \
        benchmark-smoke serve docker-build clean help

install: ## Sync the environment (incl. dev tools)
	$(UV) sync --extra dev

format: ## Auto-format with Ruff
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

lint: ## Lint with Ruff (no changes)
	$(RUN) ruff format --check src tests scripts
	$(RUN) ruff check src tests scripts

typecheck: ## Static type check with Pyright
	$(RUN) pyright

test: ## Fast CPU unit tests
	$(RUN) pytest -m "not integration and not slow and not accelerator and not multi_device"

test-integration: ## Integration tests (train/checkpoint/serve smoke)
	$(RUN) pytest -m "integration and not accelerator"

test-all: ## Everything that runs on CPU
	$(RUN) pytest -m "not accelerator"

check-export: ## Verify the repo works from tracked files only (catches ignored sources)
	bash scripts/check_clean_export.sh

train-smoke: ## 10-step CPU training run on synthetic data
	$(RUN) python scripts/train.py --config configs/train/cpu_smoke.yaml

evaluate-smoke: ## Evaluate the smoke checkpoint
	$(RUN) python scripts/evaluate.py --checkpoint artifacts/checkpoints/cpu_smoke/latest

generate-smoke: ## Greedy generation from the smoke checkpoint
	$(RUN) python scripts/generate.py --checkpoint artifacts/checkpoints/cpu_smoke/latest \
		--prompt "Once upon a time" --max-new-tokens 32 --use-kv-cache

benchmark-smoke: ## Quick benchmark sweep (CPU)
	$(RUN) python scripts/benchmark.py --config configs/benchmark/default.yaml --quick

serve: ## Serve the smoke checkpoint locally
	$(RUN) python scripts/serve.py --checkpoint artifacts/checkpoints/cpu_smoke/latest \
		--host 127.0.0.1 --port 8000

docker-build: ## Build the CPU serving image
	docker build -t jaxscale-lm:latest .

clean: ## Remove caches and build outputs (keeps artifacts/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help: ## Show targets
	@grep -E '^[a-zA-Z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "%-20s %s\n", $$1, $$2}'
