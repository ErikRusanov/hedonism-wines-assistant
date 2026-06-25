"""Tests for the indexing orchestrator with a fake store and injected embedder (I-3)."""

from collections.abc import Sequence
from pathlib import Path

from hedonism_assistant.config import Settings
from hedonism_assistant.data.index import run_index
from hedonism_assistant.vector_store.client import wine_point_id
from tests.fixtures.wines import make_wine, sample_wines


class _FakeStore:
    """Records ensure/upsert/count calls; dedupes points by id like Qdrant."""

    def __init__(self) -> None:
        self.points: dict[object, object] = {}
        self.ensure_calls = 0
        self.last_recreate: bool | None = None
        self.upsert_batch_sizes: list[int] = []

    async def ensure_collection(self, *, recreate: bool = False) -> None:
        self.ensure_calls += 1
        self.last_recreate = recreate

    async def upsert_wines(self, points: list[object]) -> None:
        self.upsert_batch_sizes.append(len(points))
        for point in points:
            self.points[point.id] = point

    async def count(self) -> int:
        return len(self.points)


class _FakeEmbedder:
    """Async embedder returning canonical vectors; records batch sizes."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0, 2.0] for text in texts]


def _write_jsonl(path: Path) -> None:
    cards = [*sample_wines(), make_wine(id="HED9", slug="no-embed")]  # last lacks embedding_text
    path.write_text("\n".join(card.model_dump_json() for card in cards) + "\n", encoding="utf-8")


def _settings(tmp_path: Path, jsonl: Path) -> Settings:
    return Settings(
        _env_file=None,
        extract_output_path=str(jsonl),
        sparse_encoder_path=str(tmp_path / "sparse.json"),
        embedding_batch_size=2,
        index_batch_size=2,
    )


async def test_run_index_reads_embeds_upserts(tmp_path: Path) -> None:
    jsonl = tmp_path / "wines.enriched.jsonl"
    _write_jsonl(jsonl)
    store, embedder = _FakeStore(), _FakeEmbedder()

    report = await run_index(_settings(tmp_path, jsonl), store=store, embed_fn=embedder)

    assert report.read == 7
    assert report.indexed == 6
    assert report.skipped_no_embedding == 1
    assert report.vector_count == 6
    # The persisted encoder is the I-5 contract artefact.
    assert (tmp_path / "sparse.json").exists()
    assert (jsonl.parent / "index_report.json").exists()


async def test_point_ids_are_deterministic_uuid5(tmp_path: Path) -> None:
    jsonl = tmp_path / "wines.enriched.jsonl"
    _write_jsonl(jsonl)
    store = _FakeStore()

    await run_index(_settings(tmp_path, jsonl), store=store, embed_fn=_FakeEmbedder())

    assert wine_point_id("HED1001") in store.points
    assert wine_point_id("HED9") not in store.points  # skipped, no embedding_text


async def test_batching_honours_configured_sizes(tmp_path: Path) -> None:
    jsonl = tmp_path / "wines.enriched.jsonl"
    _write_jsonl(jsonl)
    store, embedder = _FakeStore(), _FakeEmbedder()

    report = await run_index(_settings(tmp_path, jsonl), store=store, embed_fn=embedder)

    # 6 indexable cards: embed in batches of 2 -> 3 calls; upsert in batches of 2 -> 3 batches.
    assert embedder.batch_sizes == [2, 2, 2]
    assert store.upsert_batch_sizes == [2, 2, 2]
    assert report.batches == 3


async def test_reindex_is_idempotent(tmp_path: Path) -> None:
    jsonl = tmp_path / "wines.enriched.jsonl"
    _write_jsonl(jsonl)
    store = _FakeStore()
    settings = _settings(tmp_path, jsonl)

    await run_index(settings, store=store, embed_fn=_FakeEmbedder())
    second = await run_index(settings, store=store, embed_fn=_FakeEmbedder())

    # Same deterministic ids overwrite; the collection does not grow.
    assert second.vector_count == 6


async def test_recreate_flag_forwarded(tmp_path: Path) -> None:
    jsonl = tmp_path / "wines.enriched.jsonl"
    _write_jsonl(jsonl)
    store = _FakeStore()

    await run_index(
        _settings(tmp_path, jsonl), store=store, embed_fn=_FakeEmbedder(), recreate=True
    )
    assert store.last_recreate is True
