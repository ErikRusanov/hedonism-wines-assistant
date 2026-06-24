# hedonism-wines-assistant

Production-grade RAG assistant that answers questions about wines from the
[Hedonism Wines](https://hedonism.co.uk/wines) catalogue.

It treats the catalogue as **structured product data** and uses a hybrid
retrieval pipeline rather than long-document search:

```
query understanding (cheap LLM) → hybrid retrieval (Qdrant dense + sparse/BM25, RRF)
  → reranking (LLM listwise) → grounded generation (Claude, streamed) with citations
```

The service is stateless: no users, sessions, or chat history. Every model call —
generation, the utility model, and embeddings — goes through
[OpenRouter](https://openrouter.ai)'s OpenAI-compatible API, and vectors live in
[Qdrant](https://qdrant.tech). The whole stack runs from a single `docker compose`.

## Status

Iteration 0 (project skeleton and contracts) is in place: configuration,
structured logging, the OpenRouter client, the core Pydantic contracts, a
healthcheck, and the Docker setup. Scraping, indexing, retrieval, generation and
evaluation follow in later iterations.

## Quick start

```bash
cp .env.example .env          # then fill in OPENROUTER_API_KEY
docker compose up --build     # starts FastAPI on :8000 and Qdrant on :6333

curl http://localhost:8000/health
```

## Local development

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
source .venv/bin/activate

ruff check .                  # lint
pytest -q                     # tests

uvicorn hedonism_assistant.api.app:app --reload   # API only (needs Qdrant + .env)
```

## Data track — scraping the catalogue (I-1)

The scraper discovers products from the site's `sitemap-products.xml`, fetches
each page politely (bounded concurrency, a per-request delay and an on-disk cache
so re-runs are idempotent), and parses it into a `RawWine` written to
`data/wines.raw.jsonl`. Parsing reads `schema.org/Product` JSON-LD first and
falls back to the rendered HTML spec block; only items whose breadcrumb section
is *Wines* are kept (spirits and accessories share the sitemap). A coverage
report is written to `data/scrape_report.json`.

The catalogue is behind **Cloudflare**, which fingerprints the TLS/HTTP
handshake and 403s ordinary clients *whatever User-Agent they send*. The fetcher
uses [`curl_cffi`](https://github.com/lexiforest/curl_cffi) to impersonate a real
browser's fingerprint (`SCRAPE_IMPERSONATE`, default `chrome`), which clears the
passive check.

```bash
# Smoke run over 50 products with human-readable logs:
python -m hedonism_assistant.data.scrape --limit 50 --log-console

# Full catalogue (honours SCRAPE_* settings in .env):
python -m hedonism_assistant.data.scrape
```

Pages occasionally need JavaScript rendering; enable the optional Playwright
fallback with `uv pip install -e ".[scrape]" && playwright install chromium`,
then pass `--browser-fallback`.

> Note: the site's `robots.txt` disallows `ClaudeBot` and other AI crawlers and
> sets `ai-train=no`, while allowing the generic `User-Agent: *` with
> `search=yes`. The scraper crawls only the sitemap-advertised product pages,
> rate-limits, and caches; tune `SCRAPE_REQUEST_DELAY_SECONDS` to stay polite.

## Layout

```
src/hedonism_assistant/
  config.py            # pydantic-settings configuration
  logging_config.py    # structlog setup
  llm/openrouter.py    # OpenRouter client (chat / chat_stream / embed)
  models/              # Pydantic contracts: Wine, ParsedQuery, Chat* ...
  data/                # offline data track: sitemap → fetch → parse → wines.raw.jsonl
  api/app.py           # FastAPI app factory + /health
tests/                 # contract and smoke tests
```

## Configuration

All settings come from the environment (or a local `.env`); see `.env.example`
for the full list, including OpenRouter model slugs and Qdrant connection details.
