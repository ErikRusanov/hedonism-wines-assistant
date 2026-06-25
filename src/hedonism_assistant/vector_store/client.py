"""Qdrant hybrid wine store: connection + index-side collection management (I-3).

This is the shared foundation both data-track indexing (I-3) and serving-side
retrieval (I-5) build on. I-3 owns the index side — creating the hybrid
collection (named ``dense`` + ``sparse`` vectors), payload indexes over the
filterable :class:`WineFilters` fields, and idempotent upserts. The query-side
methods (:meth:`hybrid_query` / :meth:`dense_query`) are stubbed here and filled
in by I-5.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Final

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

if TYPE_CHECKING:
    from qdrant_client.models import ScoredPoint

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.vector_store.payload import MAX_CRITIC_SCORE_FIELD

logger = get_logger(__name__)

# Fixed namespace so a wine's string SKU maps to a stable Qdrant point UUID.
# Deterministic ids make upserts idempotent: re-indexing overwrites the same
# point instead of creating duplicates.
_WINE_ID_NAMESPACE: Final = uuid.UUID("a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d")

# Payload indexes over the hard filter fields of WineFilters. Field names match
# the frozen Wine / WineFilters contract exactly; ``grapes`` is a keyword array.
_PAYLOAD_INDEXES: Final[tuple[tuple[str, PayloadSchemaType], ...]] = (
    ("category", PayloadSchemaType.KEYWORD),
    ("color", PayloadSchemaType.KEYWORD),
    ("country", PayloadSchemaType.KEYWORD),
    ("region", PayloadSchemaType.KEYWORD),
    ("sub_region", PayloadSchemaType.KEYWORD),
    ("grapes", PayloadSchemaType.KEYWORD),
    ("vintage", PayloadSchemaType.INTEGER),
    ("price", PayloadSchemaType.FLOAT),
    ("bottle_size_ml", PayloadSchemaType.INTEGER),
    (MAX_CRITIC_SCORE_FIELD, PayloadSchemaType.FLOAT),
    ("in_bond", PayloadSchemaType.BOOL),
)


def wine_point_id(wine_id: str) -> str:
    """Deterministic Qdrant point id (UUIDv5) for a wine's string SKU."""
    return str(uuid.uuid5(_WINE_ID_NAMESPACE, wine_id))


class QdrantWineStore:
    """Async wrapper over Qdrant for the wine collection.

    Pass an explicit ``client`` to inject a fake in tests; otherwise a real
    :class:`AsyncQdrantClient` is built from settings.
    """

    __slots__ = ("_client", "_collection", "_settings")

    def __init__(self, settings: Settings, *, client: AsyncQdrantClient | None = None) -> None:
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    async def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the hybrid collection and payload indexes; idempotent.

        With ``recreate=False`` an existing collection is left untouched. With
        ``recreate=True`` it is dropped and rebuilt — a full reindex.
        """
        exists = await self._client.collection_exists(self._collection)
        if exists and not recreate:
            return
        if exists:
            await self._client.delete_collection(self._collection)

        vectors_config = {
            self._settings.qdrant_dense_vector_name: VectorParams(
                size=self._settings.embedding_dimensions,
                distance=Distance.COSINE,
            )
        }
        sparse_vectors_config = None
        if self._settings.sparse_enabled:
            sparse_vectors_config = {
                self._settings.qdrant_sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF)
            }

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        for field_name, field_schema in _PAYLOAD_INDEXES:
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        logger.info(
            "collection_ready",
            collection=self._collection,
            sparse=self._settings.sparse_enabled,
            recreated=recreate,
        )

    async def upsert_wines(self, points: list[PointStruct]) -> None:
        """Batch-upsert pre-built points into the collection."""
        await self._client.upsert(collection_name=self._collection, points=points)
        logger.info("indexed_batch", count=len(points))

    async def count(self) -> int:
        """Return the number of points in the collection."""
        result = await self._client.count(collection_name=self._collection)
        return result.count

    async def hybrid_query(
        self,
        *,
        dense_vector: list[float],
        sparse_indices: list[int] | None,
        sparse_values: list[float] | None,
        query_filter: Filter | None,
        limit: int,
        with_vectors: bool | list[str] = False,
    ) -> list[ScoredPoint]:
        """Dense + sparse retrieval fused with RRF, payload-filtered (I-5).

        Falls back to a plain dense query when the sparse channel is disabled or
        the query produced no in-vocabulary sparse terms — Qdrant rejects an
        empty sparse prefetch, so we must not send one.
        """
        if not self._settings.sparse_enabled or not sparse_indices:
            return await self.dense_query(
                dense_vector=dense_vector,
                query_filter=query_filter,
                limit=limit,
                with_vectors=with_vectors,
            )

        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        # The filter is replicated into each prefetch branch: with RRF fusion the
        # top-level query is a fusion directive, not a vector, so a top-level
        # ``query_filter`` would have nothing to filter.
        prefetch = [
            Prefetch(
                query=dense_vector,
                using=self._settings.qdrant_dense_vector_name,
                filter=query_filter,
                limit=limit,
            ),
            Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values or []),
                using=self._settings.qdrant_sparse_vector_name,
                filter=query_filter,
                limit=limit,
            ),
        ]
        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return response.points

    async def dense_query(
        self,
        *,
        dense_vector: list[float],
        query_filter: Filter | None,
        limit: int,
        with_vectors: bool | list[str] = False,
    ) -> list[ScoredPoint]:
        """Dense-only retrieval with payload filtering (I-5)."""
        response = await self._client.query_points(
            collection_name=self._collection,
            query=dense_vector,
            using=self._settings.qdrant_dense_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return response.points


@lru_cache
def get_wine_store() -> QdrantWineStore:
    """Return the cached wine store built from settings."""
    return QdrantWineStore(get_settings())
