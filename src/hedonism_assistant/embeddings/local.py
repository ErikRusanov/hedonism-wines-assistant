"""Local dense embeddings via sentence-transformers (offline).

The dense retrieval vectors are produced by a local model (default
``BAAI/bge-base-en-v1.5``) rather than OpenRouter, so indexing and retrieval run
without network access. The model is loaded lazily on first use and the blocking
``encode`` call runs off the event loop, keeping the async ``embed`` contract
identical to :meth:`OpenRouterClient.embed` so the two are interchangeable.

BGE-family models are asymmetric: passages are embedded as-is, queries get a
short instruction prefix. The index side embeds passages, so :meth:`embed`
applies no prompt; the query side (I-5) passes ``settings.embedding_query_prompt``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hedonism_assistant.config import Settings
from hedonism_assistant.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


@dataclass(slots=True)
class LocalEmbedder:
    """Embed texts with a locally-hosted sentence-transformers model.

    The model is an expensive, lazily-initialised resource cached on the instance
    (hence ``init=False``); construct one ``LocalEmbedder`` per process and reuse it.
    """

    settings: Settings
    _model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    def _load(self) -> SentenceTransformer:
        """Load (and cache) the model, importing the optional dependency lazily."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - import-guard path
            raise RuntimeError(
                "Local embeddings need sentence-transformers. Install the optional "
                'stack with: uv pip install -e ".[embed]" (or set '
                "EMBEDDING_PROVIDER=openrouter)."
            ) from exc
        device = self.settings.embedding_device or None
        logger.info(
            "embedder_loading", model=self.settings.embedding_model, device=device or "auto"
        )
        self._model = SentenceTransformer(self.settings.embedding_model, device=device)
        return self._model

    def _encode(self, texts: list[str], prompt: str | None) -> list[list[float]]:
        """Blocking encode; normalized for cosine. Runs in a worker thread."""
        vectors = self._load().encode(
            texts,
            prompt=prompt,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    async def embed(self, texts: Sequence[str], *, prompt: str | None = None) -> list[list[float]]:
        """Embed a batch of texts, one vector per input in order.

        ``prompt`` is the optional instruction prefix (used for queries, not
        passages). The index side leaves it ``None``.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts), prompt)
