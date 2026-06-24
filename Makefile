.PHONY: setup run test format

# One-time project setup: env file, virtualenv, deps, docker images, git hooks.
setup:
	test -f .env || cp .env.example .env
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	docker compose pull
	uv run pre-commit install

# Start Qdrant in Docker and run the API locally on http://127.0.0.1:8000.
run:
	docker compose up -d qdrant
	uv run uvicorn hedonism_assistant.api.app:app --reload --host 127.0.0.1 --port 8000

# Run the test suite.
test:
	uv run pytest -q

# Format and autofix with ruff.
format:
	uv run ruff format .
	uv run ruff check . --fix
