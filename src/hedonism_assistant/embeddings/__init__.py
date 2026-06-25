"""Dense embedding backends and the provider-selecting factory.

``get_embedder`` returns the async ``embed`` callable the indexer (and, later,
the retriever) use. The default is the local sentence-transformers model so the
pipeline runs offline; ``EMBEDDING_PROVIDER=openrouter`` switches back to the
OpenRouter client without changing any call sites.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from hedonism_assistant.config import EmbeddingProvider, Settings, get_settings
from hedonism_assistant.embeddings.local import LocalEmbedder

# An async callable embedding a batch of texts into one dense vector each, in
# order. Both LocalEmbedder.embed and OpenRouterClient.embed satisfy it.
type EmbedFn = Callable[[Sequence[str]], Awaitable[list[list[float]]]]


def get_embedder(settings: Settings | None = None) -> EmbedFn:
    """Return the configured dense-embedding callable."""
    settings = settings or get_settings()
    match settings.embedding_provider:
        case EmbeddingProvider.LOCAL:
            return LocalEmbedder(settings).embed
        case EmbeddingProvider.OPENROUTER:
            # Imported here so the embeddings package stays usable without the
            # OpenAI SDK configured, and to dodge a circular import at load time.
            from hedonism_assistant.llm.openrouter import get_openrouter_client

            return get_openrouter_client().embed


# A single-text async embedder for the QUERY side, with the BGE query
# instruction already applied. Used by the retriever (I-5).
type EmbedQueryFn = Callable[[str], Awaitable[list[float]]]


def get_query_embedder(settings: Settings | None = None) -> EmbedQueryFn:
    """Return a query-side embedder that applies ``embedding_query_prompt``.

    BGE is asymmetric — queries get an instruction prefix, passages do not. The
    local backend takes the prompt as a first-class ``prompt`` argument; the
    OpenRouter backend has no such parameter, so we prepend it textually (BGE
    applies the instruction as a prefix anyway, so this is equivalent).
    """
    settings = settings or get_settings()
    prompt = settings.embedding_query_prompt
    match settings.embedding_provider:
        case EmbeddingProvider.LOCAL:
            embedder = LocalEmbedder(settings)

            async def _embed_local(text: str) -> list[float]:
                vectors = await embedder.embed([text], prompt=prompt)
                return vectors[0]

            return _embed_local
        case EmbeddingProvider.OPENROUTER:
            from hedonism_assistant.llm.openrouter import get_openrouter_client

            client = get_openrouter_client()

            async def _embed_openrouter(text: str) -> list[float]:
                vectors = await client.embed([f"{prompt} {text}"])
                return vectors[0]

            return _embed_openrouter


__all__ = [
    "EmbedFn",
    "EmbedQueryFn",
    "LocalEmbedder",
    "get_embedder",
    "get_query_embedder",
]
