"""Tests for QdrantWineStore index-side behaviour against a fake client (I-3)."""

from types import SimpleNamespace

from hedonism_assistant.config import Settings
from hedonism_assistant.vector_store.client import QdrantWineStore, wine_point_id


class _FakeQdrant:
    """Minimal stand-in for AsyncQdrantClient recording index-side calls."""

    def __init__(self, *, exists: bool = False) -> None:
        self._exists = exists
        self.create_calls = 0
        self.delete_calls = 0
        self.payload_indexes: list[tuple[str, object]] = []
        self.created_vectors: dict[str, object] | None = None
        self.created_sparse: dict[str, object] | None = None
        self.points: dict[object, object] = {}

    async def collection_exists(self, collection_name: str) -> bool:
        return self._exists

    async def delete_collection(self, collection_name: str) -> None:
        self.delete_calls += 1
        self._exists = False

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: dict[str, object],
        sparse_vectors_config: dict[str, object] | None = None,
    ) -> None:
        self.create_calls += 1
        self._exists = True
        self.created_vectors = vectors_config
        self.created_sparse = sparse_vectors_config

    async def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: object
    ) -> None:
        self.payload_indexes.append((field_name, field_schema))

    async def upsert(self, collection_name: str, points: list[object]) -> None:
        for point in points:
            self.points[point.id] = point

    async def count(self, collection_name: str) -> object:
        return SimpleNamespace(count=len(self.points))


def _store(fake: _FakeQdrant) -> QdrantWineStore:
    return QdrantWineStore(Settings(_env_file=None), client=fake)


async def test_ensure_collection_builds_named_vectors_and_indexes() -> None:
    fake = _FakeQdrant(exists=False)
    settings = Settings(_env_file=None)
    await _store(fake).ensure_collection()

    assert fake.create_calls == 1
    assert set(fake.created_vectors) == {settings.qdrant_dense_vector_name}
    assert fake.created_vectors["dense"].size == settings.embedding_dimensions
    assert set(fake.created_sparse) == {settings.qdrant_sparse_vector_name}

    indexed_fields = {name for name, _ in fake.payload_indexes}
    assert {
        "category",
        "color",
        "country",
        "region",
        "sub_region",
        "grapes",
        "vintage",
        "price",
        "bottle_size_ml",
        "max_critic_score_100",
        "in_bond",
    } <= indexed_fields


async def test_sparse_vectors_skipped_when_disabled() -> None:
    fake = _FakeQdrant(exists=False)
    store = QdrantWineStore(Settings(_env_file=None, sparse_enabled=False), client=fake)
    await store.ensure_collection()
    assert fake.created_sparse is None


async def test_ensure_collection_is_idempotent() -> None:
    fake = _FakeQdrant(exists=True)
    await _store(fake).ensure_collection()
    assert fake.create_calls == 0
    assert fake.delete_calls == 0


async def test_recreate_drops_then_recreates() -> None:
    fake = _FakeQdrant(exists=True)
    await _store(fake).ensure_collection(recreate=True)
    assert fake.delete_calls == 1
    assert fake.create_calls == 1


async def test_upsert_and_count_round_trip() -> None:
    fake = _FakeQdrant(exists=True)
    store = _store(fake)
    points = [SimpleNamespace(id=wine_point_id("HED1")), SimpleNamespace(id=wine_point_id("HED2"))]
    await store.upsert_wines(points)
    assert await store.count() == 2


# Query-side behaviour (hybrid_query / dense_query, added in I-5) is exercised in
# tests/test_retriever.py against a fake client that records the query shape.
