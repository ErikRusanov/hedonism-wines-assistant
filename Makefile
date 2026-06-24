.PHONY: help setup run test format data
.DEFAULT_GOAL := help

# Data-track parameters, overridable on the command line. Examples:
#   make data LIMIT=50      # scrape + enrich the first 50 products
#   make data USE_LLM=1     # add the LLM style/pairing enrichment pass
LIMIT   ?=                  # cap records processed (empty = full run)
USE_LLM ?=                  # set to 1 to enable the LLM enrichment pass

# Show this help (targets are documented with '##' comments).
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

# One-time project setup: env file, virtualenv, deps, docker images, git hooks.
setup: ## One-time setup: .env, virtualenv, deps, docker images, git hooks
	test -f .env || cp .env.example .env
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	docker compose pull
	uv run pre-commit install

# Start Qdrant in Docker and run the API locally on http://127.0.0.1:8000.
run: ## Start Qdrant + the API locally on http://127.0.0.1:8000
	docker compose up -d qdrant
	uv run uvicorn hedonism_assistant.api.app:app --reload --host 127.0.0.1 --port 8000

# Run the test suite.
test: ## Run the test suite
	uv run pytest -q

# Format and autofix with ruff.
format: ## Format and autofix with ruff
	uv run ruff format .
	uv run ruff check . --fix

# Data track (offline): scrape the catalogue then normalize/enrich into
# wines.enriched.jsonl. Tune with LIMIT=N and USE_LLM=1 (see top of file).
data: ## Scrape + enrich the catalogue (LIMIT=N, USE_LLM=1)
	uv run python -m hedonism_assistant.data.scrape --log-console $(if $(LIMIT),--limit $(LIMIT))
	uv run python -m hedonism_assistant.data.enrich --log-console $(if $(LIMIT),--limit $(LIMIT)) $(if $(USE_LLM),--use-llm)
