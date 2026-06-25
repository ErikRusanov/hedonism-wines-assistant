"""Tests for the embedding provider factory (no model weights downloaded)."""

from hedonism_assistant.config import Settings
from hedonism_assistant.embeddings import LocalEmbedder, get_embedder


def test_get_embedder_local_returns_local_embedder_bound_method() -> None:
    settings = Settings(_env_file=None, embedding_provider="local")
    embed = get_embedder(settings)
    # The factory returns LocalEmbedder.embed bound to a fresh instance.
    assert getattr(embed, "__self__", None).__class__ is LocalEmbedder
    assert embed.__name__ == "embed"


def test_get_embedder_openrouter_returns_client_embed() -> None:
    settings = Settings(_env_file=None, embedding_provider="openrouter")
    embed = get_embedder(settings)
    from hedonism_assistant.llm.openrouter import OpenRouterClient

    assert getattr(embed, "__self__", None).__class__ is OpenRouterClient
    assert embed.__name__ == "embed"


async def test_local_embedder_empty_input_short_circuits() -> None:
    # No model load happens for an empty batch, so this stays offline.
    embedder = LocalEmbedder(Settings(_env_file=None))
    assert await embedder.embed([]) == []
