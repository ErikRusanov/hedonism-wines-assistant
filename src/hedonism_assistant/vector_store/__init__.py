"""Shared Qdrant vector-store foundation for indexing (I-3) and retrieval (I-5)."""

from hedonism_assistant.vector_store.client import (
    QdrantWineStore,
    get_wine_store,
    wine_point_id,
)
from hedonism_assistant.vector_store.payload import (
    MAX_CRITIC_SCORE_FIELD,
    build_payload,
    normalize_critic_score,
)
from hedonism_assistant.vector_store.sparse import SparseEncoder

__all__ = [
    "MAX_CRITIC_SCORE_FIELD",
    "QdrantWineStore",
    "SparseEncoder",
    "build_payload",
    "get_wine_store",
    "normalize_critic_score",
    "wine_point_id",
]
