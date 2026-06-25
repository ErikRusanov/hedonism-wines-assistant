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


__all__ = ["EmbedFn", "LocalEmbedder", "get_embedder"]
