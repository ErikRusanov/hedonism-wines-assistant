"""Tests for hybrid retrieval orchestration and the query-side store (I-5).

No network and no live Qdrant: the store is faked with canned scored points, and
the lower-level query-shape tests use a fake AsyncQdrantClient that records what
it was asked.
"""

from __future__ import annotations

from types import SimpleNamespace

from hedonism_assistant.config import RerankerKind, Settings
from hedonism_assistant.models.query import ParsedQuery, PriceRange, WineFilters
from hedonism_assistant.models.wine import RetrievedWine, WineColor
from hedonism_assistant.retrieval.rerank import NoOpReranker
from hedonism_assistant.retrieval.retriever import Retriever
from hedonism_assistant.vector_store.client import QdrantWineStore
from hedonism_assistant.vector_store.filters import build_qdrant_filter
from hedonism_assistant.vector_store.payload import build_payload
from hedonism_assistant.vector_store.sparse import SparseEncoder
from tests.fixtures.wines import sample_wines


def _scored_point(wine, score: float, vector=None):
    return SimpleNamespace(id=wine.id, score=score, payload=build_payload(wine), vector=vector)


class _FakeStore:
    """Records the hybrid_query call and returns canned points."""

    def __init__(self, points: list[object]) -> None:
        self._points = points
        self.calls: list[dict] = []

    async def hybrid_query(self, **kwargs) -> list[object]:
        self.calls.append(kwargs)
        return self._points


class _RecordingReranker:
    def __init__(self) -> None:
        self.seen_query: str | None = None

    async def rerank(self, query, candidates, *, top_k):
        self.seen_query = query
        return candidates[:top_k]


def _settings(**overrides) -> Settings:
    base = {
        "openrouter_api_key": "test",
        "reranker_kind": RerankerKind.NONE,
        "rerank_enabled": False,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


async def _capture_embed(text: str) -> list[float]:
    _capture_embed.last = text  # type: ignore[attr-defined]
    return [0.1, 0.2, 0.3]


# ---- Retriever orchestration -------------------------------------------------


async def test_reconstructs_wines_from_payloads() -> None:
    wines = sample_wines()
    points = [_scored_point(w, score=1.0 - i * 0.1) for i, w in enumerate(wines)]
    store = _FakeStore(points)
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )

    results = await retriever.retrieve(ParsedQuery(semantic_query="anything"))

    assert all(isinstance(r, RetrievedWine) for r in results)
    assert results[0].wine.id == wines[0].id
    assert results[0].score == 1.0
    assert _capture_embed.last == "anything"  # type: ignore[attr-defined]


async def test_filter_is_forwarded_to_store() -> None:
    store = _FakeStore([])
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )
    query = ParsedQuery(
        semantic_query="red Bordeaux",
        filters=WineFilters(
            color=[WineColor.RED], region=["Bordeaux"], price_range=PriceRange(max=50)
        ),
    )

    await retriever.retrieve(query)

    forwarded = store.calls[0]["query_filter"]
    keys = {c.key for c in forwarded.must}
    assert keys == {"color", "region", "price"}


class _FilterAwareStore:
    """Returns canned points only when called *without* a filter (relaxed pass)."""

    def __init__(self, unfiltered_points: list[object]) -> None:
        self._unfiltered = unfiltered_points
        self.calls: list[dict] = []

    async def hybrid_query(self, **kwargs) -> list[object]:
        self.calls.append(kwargs)
        return [] if kwargs.get("query_filter") is not None else self._unfiltered


def _filtered_query() -> ParsedQuery:
    return ParsedQuery(
        semantic_query="sweet dessert wine",
        filters=WineFilters(category=["sweet"]),  # a value no card carries
    )


async def test_empty_filtered_result_relaxes_to_filter_free_retry() -> None:
    wines = sample_wines()
    points = [_scored_point(w, score=1.0 - i * 0.1) for i, w in enumerate(wines)]
    store = _FilterAwareStore(points)
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )

    results = await retriever.retrieve(_filtered_query())

    assert [r.wine.id for r in results] == [w.id for w in wines]
    assert len(store.calls) == 2
    assert store.calls[0]["query_filter"] is not None
    assert store.calls[1]["query_filter"] is None


async def test_relaxation_disabled_keeps_empty_result() -> None:
    store = _FilterAwareStore([_scored_point(w, 1.0) for w in sample_wines()])
    retriever = Retriever(
        store,
        _capture_embed,
        NoOpReranker(),
        _settings(retrieve_relax_filters_on_empty=False),
        sparse_encoder=SparseEncoder(),
    )

    results = await retriever.retrieve(_filtered_query())

    assert results == []
    assert len(store.calls) == 1


async def test_no_relaxation_when_unfiltered_query_already_empty() -> None:
    store = _FilterAwareStore([])  # filter-free pass also empty
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )

    # No filter at all -> the relaxation guard (query_filter is not None) never trips.
    results = await retriever.retrieve(ParsedQuery(semantic_query="anything"))

    assert results == []
    assert len(store.calls) == 1


async def test_relaxation_keeps_numeric_budget_constraint() -> None:
    store = _FilterAwareStore([])
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )
    query = ParsedQuery(
        semantic_query="cheap Pauillac",
        filters=WineFilters(region=["Pauillac"], price_range=PriceRange(max=50)),
    )

    await retriever.retrieve(query)

    # Retried, but the price ceiling survives — only the categorical region is dropped.
    assert len(store.calls) == 2
    assert {c.key for c in store.calls[1]["query_filter"].must} == {"price"}


async def test_relaxation_keeps_dietary_constraint() -> None:
    # A "vegan" request must never relax into non-vegan results: the dietary flag
    # survives relaxation while the categorical region is dropped.
    store = _FilterAwareStore([])
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )
    query = ParsedQuery(
        semantic_query="vegan Burgundy",
        filters=WineFilters(region=["Burgundy"], is_vegan=True),
    )

    await retriever.retrieve(query)

    assert len(store.calls) == 2
    assert {c.key for c in store.calls[1]["query_filter"].must} == {"is_vegan"}


async def test_no_relaxation_when_only_numeric_filter_present() -> None:
    # "under £5" with nothing matching must stay empty, not relax into over-budget
    # recommendations — the relaxed filter would equal the original, so no retry.
    store = _FilterAwareStore([_scored_point(w, 1.0) for w in sample_wines()])
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )
    query = ParsedQuery(
        semantic_query="first growth Bordeaux",
        filters=WineFilters(price_range=PriceRange(max=5)),
    )

    results = await retriever.retrieve(query)

    assert results == []
    assert len(store.calls) == 1


async def test_bad_payload_is_skipped_not_raised() -> None:
    wines = sample_wines()
    good = _scored_point(wines[0], score=0.9)
    bad = SimpleNamespace(id="broken", score=0.5, payload={"not": "a wine"}, vector=None)
    store = _FakeStore([good, bad])
    retriever = Retriever(
        store, _capture_embed, NoOpReranker(), _settings(), sparse_encoder=SparseEncoder()
    )

    results = await retriever.retrieve(ParsedQuery(semantic_query="q"))
    assert [r.wine.id for r in results] == [wines[0].id]


async def test_reranker_receives_semantic_query() -> None:
    reranker = _RecordingReranker()
    points = [_scored_point(w, score=0.5) for w in sample_wines()[:2]]
    retriever = Retriever(
        _FakeStore(points),
        _capture_embed,
        reranker,
        _settings(reranker_kind=RerankerKind.LLM, rerank_enabled=True),
        sparse_encoder=SparseEncoder(),
    )

    await retriever.retrieve(ParsedQuery(semantic_query="elegant pinot"))
    assert reranker.seen_query == "elegant pinot"


async def test_missing_sparse_encoder_degrades_to_dense() -> None:
    store = _FakeStore([])
    retriever = Retriever(
        store,
        _capture_embed,
        NoOpReranker(),
        _settings(sparse_enabled=True, sparse_encoder_path="/nonexistent/encoder.json"),
    )

    await retriever.retrieve(ParsedQuery(semantic_query="q"))
    # No encoder -> no sparse vector forwarded; store falls back to dense.
    assert store.calls[0]["sparse_indices"] is None


# ---- Query-side store shape (QdrantWineStore) --------------------------------


class _FakeQdrant:
    """Records query_points kwargs and returns canned scored points."""

    def __init__(self, points: list[object]) -> None:
        self._points = points
        self.last_kwargs: dict | None = None

    async def query_points(self, **kwargs) -> object:
        self.last_kwargs = kwargs
        return SimpleNamespace(points=self._points)


def _store(fake: _FakeQdrant, **overrides) -> QdrantWineStore:
    return QdrantWineStore(Settings(_env_file=None, **overrides), client=fake)


async def test_hybrid_query_replicates_filter_into_both_prefetches() -> None:
    fake = _FakeQdrant([])
    store = _store(fake)
    query_filter = build_qdrant_filter(WineFilters(color=[WineColor.RED]))

    await store.hybrid_query(
        dense_vector=[0.1, 0.2],
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.5],
        query_filter=query_filter,
        limit=40,
    )

    prefetch = fake.last_kwargs["prefetch"]
    assert len(prefetch) == 2
    assert {p.using for p in prefetch} == {"dense", "sparse"}
    assert all(p.filter is query_filter for p in prefetch)
    # Fusion happens at the top level; no top-level query_filter in hybrid mode.
    assert fake.last_kwargs.get("query_filter") is None


async def test_hybrid_query_falls_back_to_dense_without_sparse_terms() -> None:
    fake = _FakeQdrant([])
    store = _store(fake)

    await store.hybrid_query(
        dense_vector=[0.1, 0.2],
        sparse_indices=[],
        sparse_values=[],
        query_filter=None,
        limit=40,
    )

    # Dense path: top-level query + query_filter, no prefetch.
    assert "prefetch" not in fake.last_kwargs
    assert fake.last_kwargs["using"] == "dense"


async def test_hybrid_query_dense_only_when_sparse_disabled() -> None:
    fake = _FakeQdrant([])
    store = _store(fake, sparse_enabled=False)

    await store.hybrid_query(
        dense_vector=[0.1],
        sparse_indices=[1],
        sparse_values=[0.9],
        query_filter=None,
        limit=10,
    )
    assert "prefetch" not in fake.last_kwargs


async def test_dense_query_uses_top_level_query_filter() -> None:
    fake = _FakeQdrant([])
    store = _store(fake)
    sentinel = object()

    await store.dense_query(dense_vector=[0.1], query_filter=sentinel, limit=5)  # type: ignore[arg-type]
    assert fake.last_kwargs["query_filter"] is sentinel
    assert fake.last_kwargs["using"] == "dense"
