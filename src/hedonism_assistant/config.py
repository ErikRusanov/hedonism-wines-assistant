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


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
