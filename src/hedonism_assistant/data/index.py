"""CLI orchestrator for indexing canonical wine cards into Qdrant (I-3).

Pipeline: read ``wines.enriched.jsonl`` (one serialised :class:`Wine` per line,
each carrying a precomputed ``embedding_text``) -> fit and persist the shared
sparse encoder -> ensure the hybrid collection exists -> embed the passports in
batches via the configured embedder (local by default) -> upsert dense+sparse
points with deterministic ids. A report mirrors the extract report so a run's
outcome is visible at a glance.

Run it as a module::

    python -m hedonism_assistant.data.index --log-console
    python -m hedonism_assistant.data.index --recreate   # full reindex

Point ids are deterministic (UUIDv5 of the SKU), so re-running is idempotent:
the same cards overwrite their own points instead of accumulating duplicates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import ValidationError

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.embeddings import EmbedFn, get_embedder
from hedonism_assistant.logging_config import configure_logging, get_logger
from hedonism_assistant.models.wine import Wine
from hedonism_assistant.vector_store.client import QdrantWineStore, get_wine_store, wine_point_id
from hedonism_assistant.vector_store.payload import build_payload
from hedonism_assistant.vector_store.sparse import SparseEncoder

logger = get_logger(__name__)


@dataclass(slots=True)
class IndexReport:
    """Counters for one indexing run, persisted to ``data/index_report.json``."""

    read: int = 0
    indexed: int = 0
    skipped_no_embedding: int = 0
    batches: int = 0
    collection: str = ""
    vector_count: int = 0


def _read_wines(path: Path, report: IndexReport, limit: int | None) -> list[Wine]:
    """Load canonical cards from JSONL, counting reads and skipping blank lines."""
    wines: list[Wine] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if limit is not None and report.read >= limit:
                break
            report.read += 1
            try:
                wines.append(Wine.model_validate_json(stripped))
            except ValidationError as exc:
                logger.warning("enriched_parse_failed", error=str(exc))
    return wines


def _chunks[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of ``items`` of at most ``size`` elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_point(
    wine: Wine,
    dense: list[float],
    encoder: SparseEncoder,
    settings: Settings,
) -> object:
    """Assemble a Qdrant point with named dense (+ optional sparse) vectors."""
    # Imported lazily so the module stays importable without a live Qdrant.
    from qdrant_client.models import PointStruct, SparseVector

    vector: dict[str, object] = {settings.qdrant_dense_vector_name: dense}
    if settings.sparse_enabled:
        indices, values = encoder.encode(wine.embedding_text or "")
        vector[settings.qdrant_sparse_vector_name] = SparseVector(indices=indices, values=values)
    return PointStruct(
        id=wine_point_id(wine.id),
        vector=vector,
        payload=build_payload(wine),
    )


async def run_index(
    settings: Settings,
    *,
    limit: int | None = None,
    recreate: bool = False,
    store: QdrantWineStore | None = None,
    embed_fn: EmbedFn | None = None,
) -> IndexReport:
    """Execute a full indexing pass; writes the report file and returns it."""
    report = IndexReport(collection=settings.qdrant_collection)
    wines = _read_wines(Path(settings.extract_output_path), report, limit)
    logger.info("read_done", read=report.read)

    # Fit and persist the sparse encoder over every available passport — the
    # query side (I-5) must load this exact file or the sparse channel skews.
    indexable = [w for w in wines if w.embedding_text]
    report.skipped_no_embedding = len(wines) - len(indexable)
    encoder = SparseEncoder.fit(w.embedding_text for w in indexable if w.embedding_text)
    encoder.save(settings.sparse_encoder_path)

    store = store or get_wine_store()
    embed_fn = embed_fn or get_embedder(settings)
    await store.ensure_collection(recreate=recreate)

    points: list[object] = []
    for chunk in _chunks(indexable, settings.embedding_batch_size):
        texts = [w.embedding_text or "" for w in chunk]
        vectors = await embed_fn(texts)
        for wine, dense in zip(chunk, vectors, strict=True):
            points.append(_build_point(wine, dense, encoder, settings))

    for batch in _chunks(points, settings.index_batch_size):
        await store.upsert_wines(list(batch))
        report.batches += 1
    report.indexed = len(points)
    report.vector_count = await store.count()

    report_path = Path(settings.extract_output_path).with_name("index_report.json")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info("index_done", **asdict(report))
    return report


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Layer CLI overrides on top of the environment-backed settings."""
    overrides: dict[str, object] = {}
    if args.input is not None:
        overrides["extract_output_path"] = args.input
    return get_settings().model_copy(update=overrides)


def _print_summary(report: IndexReport) -> None:
    print("\nIndex summary")
    print("-------------")
    print(f"  read                : {report.read}")
    print(f"  indexed             : {report.indexed}")
    print(f"  skipped (no embed)  : {report.skipped_no_embedding}")
    print(f"  upsert batches      : {report.batches}")
    print(f"  collection          : {report.collection}")
    print(f"  vector count        : {report.vector_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index canonical wine cards into Qdrant (I-3).")
    parser.add_argument("--input", help="Input enriched JSONL path.")
    parser.add_argument("--limit", type=int, help="Cap the number of cards indexed.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the collection (full reindex).",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="Human-readable logs instead of JSON.",
    )
    args = parser.parse_args()

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_output=not args.log_console)
    report = asyncio.run(run_index(settings, limit=args.limit, recreate=args.recreate))
    _print_summary(report)


if __name__ == "__main__":
    main()
