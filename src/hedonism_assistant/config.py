"""Application configuration loaded from the environment via pydantic-settings."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class EmbeddingProvider(StrEnum):
    """Backend that produces dense vectors. ``LOCAL`` keeps the pipeline offline."""

    LOCAL = "local"
    OPENROUTER = "openrouter"


class RerankerKind(StrEnum):
    """Reranking backend. ``LLM`` listwise (via the cheap utility model) is the
    default — it adds no new service. ``NONE`` disables reranking entirely.
    Cohere/Voyage rerankers are future drop-ins behind the same protocol.
    """

    LLM = "llm"
    NONE = "none"


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

    # NoDecode: keep the raw env string so the CSV validator below handles it,
    # instead of pydantic-settings attempting to JSON-decode the value first.
    generation_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    utility_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "hedonism_wines"

    # Embeddings (local by default; runs fully offline). Dense vectors come from a
    # local sentence-transformers model rather than OpenRouter, so indexing works
    # without network access. Generation and the utility model stay on OpenRouter.
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # local model id (HF) or OpenRouter slug
    embedding_dimensions: int = 768  # bge-base (frozen index<->query contract)
    embedding_device: str = ""  # "" = auto-detect (mps/cuda/cpu)
    # BGE asks for a query instruction on the query side only; passages get none.
    # The index side (here) embeds passages; I-5's query side honors this prompt.
    embedding_query_prompt: str = "Represent this sentence for searching relevant passages:"
    embedding_batch_size: int = 64  # texts per embed call

    # Indexing (Qdrant collection). The vector names and sparse toggle are the
    # frozen I-3<->I-5 contract: the query side (I-5) must read the same names and
    # load the same persisted sparse encoder, or retrieval skews.
    qdrant_dense_vector_name: str = "dense"  # frozen, I-5
    qdrant_sparse_vector_name: str = "sparse"  # frozen, I-5
    sparse_enabled: bool = True  # build sparse/BM25 vectors (shared with I-5)
    sparse_encoder_path: str = "data/sparse_encoder.json"  # persisted fitted IDF (reused by I-5)
    index_batch_size: int = 128  # points per Qdrant upsert

    # App
    log_level: str = "INFO"
    log_json: bool = True
    request_timeout_seconds: float = 60.0
    max_retries: int = 3

    # Serving (I-7). CORS origins for the chat page / API. Permissive by default
    # for the demo; tighten to specific origins in production (I-9). Accepts a
    # comma-separated string in the environment (see the CSV validator below).
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    # Data extraction (offline). The catalogue is no longer scraped: product-page
    # HTML is captured by hand (see data/chrome_capture_prompt.md) and dropped as
    # <slug>.html files in the cache below. `extract` parses + normalizes them into
    # canonical Wine cards. The base URL only synthesizes a page URL from a slug.
    catalogue_base_url: str = "https://hedonism.co.uk"
    html_input_dir: str = "data/cache/html"  # captured <slug>.html files
    extract_output_path: str = "data/wines.enriched.jsonl"  # canonical Wine cards
    # How much of the (copyrighted) tasting note to fold into the embedding text.
    # Bounded to keep dense vectors focused and to store the source sparingly.
    embedding_text_notes_chars: int = 600

    # Query understanding (self-query)
    query_parsing_enabled: bool = True
    query_parse_temperature: float = 0.0

    # Retrieval (I-5). Hybrid dense+sparse -> RRF fusion -> rerank -> optional MMR.
    # Every stage is toggle-gated for tuning; whether the sparse channel is used at
    # all is governed by the shared ``sparse_enabled`` toggle above (no separate
    # hybrid flag), so index and query stay in lock-step.
    retrieve_top_n: int = 40  # candidates fetched from Qdrant before reranking
    rerank_top_k: int = 8  # final result count after rerank/MMR
    rerank_enabled: bool = True
    reranker_kind: RerankerKind = RerankerKind.LLM
    rerank_temperature: float = 0.0  # deterministic listwise ordering
    mmr_enabled: bool = False  # diversify the final list (needs candidate vectors)
    mmr_lambda: float = 0.5  # MMR relevance/diversity trade-off in [0, 1]

    # Answer generation (I-6). The retrieved cards are folded into a grounded
    # prompt and the generation model streams the answer; citations are derived
    # from inline [n] markers, so these toggles only shape the prompt and prose.
    generation_temperature: float = 0.3  # a little warmth for natural prose, still grounded
    generation_context_max_wines: int = 8  # cards folded into the prompt context
    generation_note_chars: int = 400  # per-card tasting-note cap (token budget + injection surface)
    generation_max_suggestions: int = 3  # follow-ups offered on out-of-scope / empty retrieval
    # Where to send users who ask about non-wine drinks. We only know wine, so the
    # "other drinks" guardrail redirects them to Hedonism's spirits range.
    spirits_url: str = "https://hedonism.co.uk/spirits"

    # Evaluation (I-8). The golden-set regression harness runs the live pipeline
    # over data/golden_set.jsonl and scores retrieval (hit@k, MRR) plus answer
    # quality via the utility model as LLM-judge (faithfulness, answer relevancy).
    # Thresholds gate the run: ``make eval`` exits non-zero when a mean falls
    # below its bound. The judge reuses ``utility_model`` (no separate slug).
    golden_set_path: str = "data/golden_set.jsonl"
    eval_report_path: str = "data/eval_report.json"
    eval_judge_enabled: bool = True  # call the LLM judge; off = retrieval-only
    eval_judge_temperature: float = 0.0  # deterministic judging
    eval_min_hit_at_k: float = 0.80
    eval_min_mrr: float = 0.60
    eval_min_faithfulness: float = 0.85
    eval_min_answer_relevancy: float = 0.70

    @field_validator(
        "generation_fallback_models",
        "utility_fallback_models",
        "cors_allow_origins",
        mode="before",
    )
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
