"""Hybrid retrieval orchestration (I-5) — the merge point of both tracks.

Given a :class:`ParsedQuery` (produced upstream by query understanding, I-4) and
the live Qdrant index (built by I-3), the retriever:

1. translates the hard filters into a Qdrant payload filter;
2. dense-embeds the semantic query (with the BGE query prompt) and sparse-encodes
   it with the *same persisted encoder* used at index time;
3. runs a hybrid (dense+sparse, RRF-fused) Qdrant query, payload-filtered;
4. rebuilds :class:`RetrievedWine` cards straight from the point payloads;
5. reranks down to the final top-K;
6. optionally diversifies with MMR.

It receives an already-parsed query — parsing is a separate stage — and never
raises on a single bad payload or a missing sparse encoder; it degrades instead.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import ValidationError

from hedonism_assistant.config import RerankerKind, Settings, get_settings
from hedonism_assistant.embeddings import EmbedQueryFn, get_query_embedder
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.query import ParsedQuery
from hedonism_assistant.models.wine import RetrievedWine, Wine
from hedonism_assistant.retrieval.mmr import ScoredCandidate, Vector, mmr_select
from hedonism_assistant.retrieval.rerank import Reranker, get_reranker
from hedonism_assistant.vector_store.client import QdrantWineStore, get_wine_store
from hedonism_assistant.vector_store.filters import build_qdrant_filter
from hedonism_assistant.vector_store.sparse import SparseEncoder

logger = get_logger(__name__)


class Retriever:
    """Compose hybrid retrieval, reranking and optional MMR into one call."""

    __slots__ = (
        "_store",
        "_embed_query",
        "_reranker",
        "_settings",
        "_sparse_encoder",
        "_sparse_ready",
    )

    def __init__(
        self,
        store: QdrantWineStore,
        embed_query: EmbedQueryFn,
        reranker: Reranker,
        settings: Settings,
        *,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self._store = store
        self._embed_query = embed_query
        self._reranker = reranker
        self._settings = settings
        self._sparse_encoder = sparse_encoder
        # "Resolved" means we've either got the encoder or established it's
        # missing — an injected one is resolved up front; otherwise it loads
        # lazily on first use and a missing file is remembered (no retry storm).
        self._sparse_ready = sparse_encoder is not None

    def _resolve_encoder(self) -> SparseEncoder | None:
        """Lazily load (and cache) the persisted sparse encoder, or ``None``."""
        if not self._sparse_ready:
            try:
                self._sparse_encoder = SparseEncoder.load(self._settings.sparse_encoder_path)
            except FileNotFoundError:
                logger.warning(
                    "sparse_encoder_missing",
                    path=self._settings.sparse_encoder_path,
                    detail="falling back to dense-only retrieval",
                )
                self._sparse_encoder = None
            self._sparse_ready = True
        return self._sparse_encoder

    def _sparse_encode(self, text: str) -> tuple[list[int], list[float]] | None:
        """Encode the query with the persisted encoder; ``None`` if unavailable."""
        if not self._settings.sparse_enabled:
            return None
        encoder = self._resolve_encoder()
        return encoder.encode(text) if encoder is not None else None

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedWine]:
        """Retrieve, rerank and (optionally) diversify wines for a parsed query."""
        query_filter = build_qdrant_filter(query.filters)
        dense_vector = await self._embed_query(query.semantic_query)

        encoded = self._sparse_encode(query.semantic_query)
        sparse_indices, sparse_values = encoded if encoded is not None else (None, None)

        want_vectors: bool | list[str] = (
            [self._settings.qdrant_dense_vector_name] if self._settings.mmr_enabled else False
        )
        points = await self._store.hybrid_query(
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            query_filter=query_filter,
            limit=self._settings.retrieve_top_n,
            with_vectors=want_vectors,
        )

        candidates: list[RetrievedWine] = []
        vectors_by_id: dict[str, Vector] = {}
        for point in points:
            try:
                wine = Wine.model_validate(point.payload)
            except ValidationError as exc:
                logger.warning("payload_parse_failed", point_id=str(point.id), error=str(exc))
                continue
            candidates.append(RetrievedWine(wine=wine, score=point.score or 0.0))
            if self._settings.mmr_enabled:
                vector = self._dense_vector_of(point)
                if vector is not None:
                    vectors_by_id[wine.id] = vector

        top_k = self._settings.rerank_top_k
        if self._settings.rerank_enabled and self._settings.reranker_kind != RerankerKind.NONE:
            candidates = await self._reranker.rerank(query.semantic_query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        if self._settings.mmr_enabled:
            candidates = self._apply_mmr(candidates, vectors_by_id)

        return candidates

    def _apply_mmr(
        self, candidates: list[RetrievedWine], vectors_by_id: dict[str, Vector]
    ) -> list[RetrievedWine]:
        """Diversify with MMR; skip (and log) if any candidate lacks a vector."""
        paired: list[ScoredCandidate] = []
        for candidate in candidates:
            vector = vectors_by_id.get(candidate.wine.id)
            if vector is None:
                logger.warning("mmr_skipped", detail="missing candidate vectors")
                return candidates
            paired.append((candidate, vector))
        return mmr_select(
            paired, lambda_=self._settings.mmr_lambda, top_k=self._settings.rerank_top_k
        )

    def _dense_vector_of(self, point: object) -> Vector | None:
        """Pull the named dense vector off a scored point, if present."""
        vector = getattr(point, "vector", None)
        if isinstance(vector, dict):
            dense = vector.get(self._settings.qdrant_dense_vector_name)
            return dense if isinstance(dense, list) else None
        return vector if isinstance(vector, list) else None


@lru_cache
def get_retriever() -> Retriever:
    """Return the cached retriever built from settings and shared singletons."""
    settings = get_settings()
    return Retriever(
        store=get_wine_store(),
        embed_query=get_query_embedder(settings),
        reranker=get_reranker(settings),
        settings=settings,
    )
