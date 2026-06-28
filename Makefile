.PHONY: help setup embed-setup run test format data index eval deploy
.DEFAULT_GOAL := help

# Data-track parameters, overridable on the command line. Examples:
#   make data                    # extract all captured HTML into enriched cards
#   make data LIMIT=50           # extract the first 50 HTML files
#   make index                   # embed + index the enriched cards into Qdrant
#   make index RECREATE=1        # drop + rebuild the Qdrant collection
#   make eval                    # run the golden-set regression harness
#   make eval NOJUDGE=1          # retrieval metrics only (network-light)
LIMIT   ?=                  # cap records processed (empty = full run)
RECREATE ?=                 # set to 1 to drop+rebuild the Qdrant collection (index)
NOJUDGE ?=                  # set to 1 to skip the LLM judge (eval, retrieval-only)

# Show this help (targets are documented with '##' comments).
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# One-time project setup: env file, virtualenv, deps, docker images, git hooks.
setup: ## One-time setup: .env, virtualenv, deps, docker images, git hooks
	test -f .env || cp .env.example .env
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	docker compose pull
	uv run pre-commit install

# Install the optional local-embedding stack (torch + sentence-transformers).
embed-setup: ## One-time embedding setup: install the local sentence-transformers stack
	uv pip install -e ".[embed]"

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

# Data track (offline): extract canonical Wine cards from the captured product
# HTML (data/cache/html/*.html) into wines.enriched.jsonl. Deterministic, no network.
data: ## Extract captured HTML into enriched cards — LIMIT
	uv run python -m hedonism_assistant.data.extract --log-console \
		$(if $(LIMIT),--limit $(LIMIT))

# Index the enriched cards (wines.enriched.jsonl) into Qdrant as a hybrid
# dense+sparse collection. Starts Qdrant in Docker first. RECREATE=1 drops and
# rebuilds the collection; LIMIT caps the number of cards.
index: ## Index enriched cards into Qdrant — LIMIT/RECREATE
	docker compose up -d qdrant
	uv run python -m hedonism_assistant.data.index --log-console \
		$(if $(LIMIT),--limit $(LIMIT)) $(if $(RECREATE),--recreate)

# Run the golden-set regression harness against the live Qdrant index: retrieval
# metrics (hit@k, MRR) + LLM-judge answer quality, writing data/eval_report.json
# with pass/fail thresholds (non-zero exit on a miss). Unlike data/index, the
# judge needs network + a funded OpenRouter key; NOJUDGE=1 is retrieval-only.
eval: ## Run the golden-set eval harness against live Qdrant — LIMIT/NOJUDGE
	docker compose up -d qdrant
	uv run python -m hedonism_assistant.eval.run --log-console \
		$(if $(LIMIT),--limit $(LIMIT)) $(if $(NOJUDGE),--no-judge)

# Build the serving image locally and ship the whole stack to the production
# server over SSH (no registry): app image + qdrant + caddy, the local Qdrant
# index, and .env.prod. Interactive; AUTO_YES=1 skips the prompts. Override the
# target with SERVER_IP=, SSH_USER=, DOMAIN=, etc. See scripts/deploy.sh.
deploy: ## Build locally + ship the full stack to the prod server over SSH
	bash scripts/deploy.sh

