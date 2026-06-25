"""Live end-to-end smoke test for I-5 retrieval (gated on ``QDRANT_LIVE``).

This is the acceptance check: a query with hard filters returns the right wines
with the filters *genuinely applied* at the index, not just semantically. It
needs a running Qdrant and the local embedding stack (``.[embed]``), so it is
skipped unless ``QDRANT_LIVE=1``.

It runs offline of OpenRouter: the query is constructed directly (no LLM parse)
and reranking is disabled, so only local embeddings + Qdrant are exercised.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hedonism_assistant.config import RerankerKind, get_settings
from hedonism_assistant.data.index import run_index
from hedonism_assistant.embeddings import get_query_embedder
from hedonism_assistant.models.query import ParsedQuery, PriceRange, WineFilters
from hedonism_assistant.models.wine import WineColor
from hedonism_assistant.retrieval.rerank import NoOpReranker
from hedonism_assistant.retrieval.retriever import Retriever
from hedonism_assistant.vector_store.client import QdrantWineStore
from tests.fixtures.wines import sample_wines

pytestmark = pytest.mark.skipif(
    not os.getenv("QDRANT_LIVE"),
    reason="needs a live Qdrant and local embeddings; set QDRANT_LIVE=1 to run",
)


async def test_filters_are_applied_at_the_index(tmp_path: Path) -> None:
    jsonl = tmp_path / "wines.jsonl"
    jsonl.write_text("\n".join(w.model_dump_json() for w in sample_wines()), encoding="utf-8")
    settings = get_settings().model_copy(
        update={
            "extract_output_path": str(jsonl),
            "sparse_encoder_path": str(tmp_path / "sparse_encoder.json"),
            "qdrant_collection": "hedonism_wines_test_i5",
            "reranker_kind": RerankerKind.NONE,
            "rerank_enabled": False,
        }
    )

    store = QdrantWineStore(settings)
    await run_index(settings, recreate=True, store=store)

    retriever = Retriever(store, get_query_embedder(settings), NoOpReranker(), settings)
    query = ParsedQuery(
        semantic_query="white Burgundy",
        filters=WineFilters(
            color=[WineColor.WHITE], region=["Burgundy"], price_range=PriceRange(max=50)
        ),
    )

    results = await retriever.retrieve(query)
    ids = {r.wine.id for r in results}

    # Chablis Droin 2022 (white Burgundy, £45) satisfies the filter...
    assert "HED1002" in ids
    # ...and the filters genuinely exclude the £320 red Bordeaux and every other
    # colour/region/price mismatch — not merely rank them lower.
    assert ids == {"HED1002"}
    assert all(
        r.wine.color is WineColor.WHITE and r.wine.region == "Burgundy" and r.wine.price <= 50
        for r in results
    )

    # Clean up the throwaway collection.
    await store.client.delete_collection(settings.qdrant_collection)
