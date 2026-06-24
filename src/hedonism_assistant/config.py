"""Application configuration loaded from the environment via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for runtime configuration.

    Values are read from environment variables (and a local ``.env`` file in
    development). See ``.env.example`` for the full list with descriptions.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    generation_model: str = "anthropic/claude-opus-4-8"
    utility_model: str = "anthropic/claude-haiku-4-5"
    embedding_model: str = "openai/text-embedding-3-large"

    # NoDecode: keep the raw env string so the CSV validator below handles it,
    # instead of pydantic-settings attempting to JSON-decode the value first.
    generation_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    utility_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "hedonism_wines"

    # App
    log_level: str = "INFO"
    log_json: bool = True
    request_timeout_seconds: float = 60.0
    max_retries: int = 3

    # Scraper (data track, offline). The catalogue sits behind Cloudflare, so we
    # fetch every page through a real Playwright Chromium (the only thing that
    # gets past the bot check). Every crawl knob lives here.
    scrape_base_url: str = "https://hedonism.co.uk"
    # User-Agent the browser presents.
    scrape_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    scrape_request_delay_seconds: float = 1.0  # polite gap between requests
    scrape_max_concurrency: int = 4
    scrape_timeout_seconds: float = 30.0
    scrape_max_retries: int = 3
    scrape_cache_dir: str = "data/cache"  # raw HTML, for idempotent re-runs
    scrape_output_path: str = "data/wines.raw.jsonl"
    # Keep only catalogue items whose breadcrumb section is "Wines" (drops
    # spirits, accessories and books that share the listing).
    scrape_wines_only: bool = True
    # Optional cap on the number of products fetched, for quick smoke runs.
    scrape_max_products: int | None = None

    scrape_browser_headless: bool = True
    # When Cloudflare shows an interactive challenge, point this at a Chrome
    # profile dir so the cf_clearance cookie persists across runs (solve once).
    scrape_browser_user_data_dir: str = ""
    scrape_browser_wait_until: str = "domcontentloaded"  # domcontentloaded|load|networkidle

    # Normalization & enrichment (data track, offline). Turns the permissive
    # scrape output into canonical Wine cards ready for indexing.
    enrich_input_path: str = "data/wines.raw.jsonl"
    enrich_output_path: str = "data/wines.enriched.jsonl"
    # How much of the (copyrighted) tasting note to fold into the embedding text.
    # Bounded to keep dense vectors focused and to store the source sparingly.
    embedding_text_notes_chars: int = 600
    # Optional LLM pass: fill a missing colour and add style/food-pairing tags
    # with the cheap utility model. Off by default so the pipeline runs fully
    # offline and deterministically; turn on to enrich.
    enrich_use_llm: bool = False
    enrich_llm_temperature: float = 0.0
    enrich_llm_concurrency: int = 4
    # Cap on style_tags / food_pairings kept per card after the LLM pass, so a
    # chatty model cannot bloat the payload or the embedding text.
    enrich_max_tags: int = 6

    # Query understanding (self-query)
    query_parsing_enabled: bool = True
    query_parse_temperature: float = 0.0

    @field_validator("generation_fallback_models", "utility_fallback_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated strings for list-valued settings."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("scrape_max_products", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        """Treat an empty/blank env value as unset, so ``SCRAPE_MAX_PRODUCTS=``
        means "scrape the whole catalogue" instead of failing int parsing."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
