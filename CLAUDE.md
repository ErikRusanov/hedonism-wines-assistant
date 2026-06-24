# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language policy (strict)

All project artefacts — code, comments, docstrings, identifiers, commit messages,
documentation, log messages — MUST be written in English. No Russian anywhere in
the repository. (Conversation with the repository owner happens in Russian, but
nothing Russian is ever committed.)

## Commands

The project uses `uv` and a `src/` layout. A virtualenv lives in `.venv`.

```bash
uv venv --python 3.12          # create the virtualenv (once)
uv pip install -e ".[dev]"     # install runtime + dev dependencies

source .venv/bin/activate
ruff check .                   # lint
ruff check . --fix             # lint + autofix
pytest -q                      # run the full test suite
pytest tests/test_models.py -q # run one test file
pytest -k parsed_query         # run tests matching an expression

docker compose up --build      # run FastAPI + Qdrant together
uvicorn hedonism_assistant.api.app:app --reload   # run the API alone (needs Qdrant + .env)
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so tests import
`hedonism_assistant` without an editable install, but Docker/uvicorn need the
package installed.

## Architecture

This is a production-grade RAG service answering questions about wines from the
Hedonism catalogue. It is **stateless** — no users, sessions, or chat history.
The full design and iteration plan live in `.claude/plans/` (gitignored, Russian).

The catalogue is **structured product data**, so retrieval is a hybrid pipeline,
not long-document search:

```
query-parse (cheap LLM) → hybrid retrieve (Qdrant dense + sparse/BM25, RRF) →
rerank (LLM listwise) → grounded generation (Claude, streamed) with citations
```

Two tracks run off the shared contracts:
- **Data (offline):** scrape `hedonism.co.uk/wines` → normalize/enrich → index into Qdrant.
- **Serving (online):** FastAPI endpoints (`/chat` SSE, `/search`, `/health`) over a static chat page.

### Single provider: OpenRouter

Every model call — generation (Claude Opus), the cheap utility model
(query parsing / reranking / eval judge) and embeddings — goes through
OpenRouter's OpenAI-compatible API. `llm/openrouter.py` is the only place that
talks to the raw OpenAI SDK; everything else uses `OpenRouterClient`
(`chat`, `chat_stream`, `embed`) which owns model selection, fallback chains and
tenacity retries. Do not import `openai` elsewhere.

### Contracts are the backbone

`models/` holds the Pydantic contracts that both tracks depend on. Change these
deliberately — they are the integration seam:
- `Wine` — canonical product card produced by the data track, consumed by indexing/serving.
- `RetrievedWine` — `Wine` plus fusion/rerank scores.
- `ParsedQuery` — output of query understanding: `semantic_query` + hard `WineFilters` + `intent`.
  Filters map directly onto Qdrant payload-index filters (the reason "under £50",
  "Bordeaux", "2015" must become filters, not free text).
- `ChatRequest` / `ChatResponse` — the stateless API surface, with grounded `WineCitation`s.

### Configuration & logging

`config.py` is the single config source (`pydantic-settings`, `.env`); get it via
`get_settings()`. `logging_config.py` configures structlog (JSON in prod,
console in dev) — use `get_logger(__name__)`, never bare `print`.

## Conventions

- Python ≥ 3.12; prefer modern typing (`X | None`, `StrEnum`, `list[...]`).
- Ruff is the linter/formatter (line length 100); keep `ruff check .` clean.
- New tunable behaviour should be a config toggle in `config.py`, not a hardcoded constant —
  the retrieval pipeline (top-N/K, reranker, MMR, fusion) is meant to be tuned.
