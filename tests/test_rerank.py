"""Unit tests for the listwise reranker (I-5).

The network is never touched: ``client.chat`` is replaced with a stub, so only
the reranker's parsing, reordering and resilience logic is tested.
"""

from __future__ import annotations

import json

from hedonism_assistant.config import Settings
from hedonism_assistant.llm.openrouter import OpenRouterClient
from hedonism_assistant.models.wine import RetrievedWine
from hedonism_assistant.retrieval.rerank import LLMListwiseReranker, NoOpReranker
from tests.fixtures.wines import sample_wines


def _candidates(n: int | None = None) -> list[RetrievedWine]:
    wines = sample_wines()
    if n is not None:
        wines = wines[:n]
    # Descending fusion scores so original order is meaningful.
    return [RetrievedWine(wine=w, score=1.0 - i * 0.1) for i, w in enumerate(wines)]


def _reranker_returning(payload: object) -> LLMListwiseReranker:
    settings = Settings(openrouter_api_key="test")
    client = OpenRouterClient(settings)

    async def fake_chat(messages, **kwargs) -> str:
        if isinstance(payload, Exception):
            raise payload
        return payload if isinstance(payload, str) else json.dumps(payload)

    client.chat = fake_chat  # type: ignore[method-assign]
    return LLMListwiseReranker(client, settings)


async def test_reorders_and_sets_rerank_score() -> None:
    reranker = _reranker_returning(
        {"ranking": [{"index": 2, "score": 0.9}, {"index": 0, "score": 0.7}]}
    )
    candidates = _candidates(3)

    result = await reranker.rerank("a query", candidates, top_k=3)

    # Ranked first two by the model, then the omitted candidate (index 1) appended.
    assert [r.wine.id for r in result] == [
        candidates[2].wine.id,
        candidates[0].wine.id,
        candidates[1].wine.id,
    ]
    assert result[0].rerank_score == 0.9
    assert result[1].rerank_score == 0.7
    assert result[2].rerank_score is None


async def test_truncates_to_top_k() -> None:
    reranker = _reranker_returning(
        {"ranking": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.5}]}
    )
    result = await reranker.rerank("q", _candidates(4), top_k=2)
    assert len(result) == 2


async def test_malformed_json_preserves_input_order() -> None:
    reranker = _reranker_returning("not json at all")
    candidates = _candidates(3)
    result = await reranker.rerank("q", candidates, top_k=3)
    assert [r.wine.id for r in result] == [c.wine.id for c in candidates]
    assert all(r.rerank_score is None for r in result)


async def test_chat_error_preserves_input_order() -> None:
    reranker = _reranker_returning(RuntimeError("upstream down"))
    candidates = _candidates(3)
    result = await reranker.rerank("q", candidates, top_k=2)
    assert [r.wine.id for r in result] == [c.wine.id for c in candidates[:2]]


async def test_out_of_range_and_duplicate_indices_dropped() -> None:
    reranker = _reranker_returning(
        {
            "ranking": [
                {"index": 99, "score": 1.0},  # out of range
                {"index": 1, "score": 0.8},
                {"index": 1, "score": 0.4},  # duplicate
            ]
        }
    )
    candidates = _candidates(3)
    result = await reranker.rerank("q", candidates, top_k=3)

    # Only index 1 is valid; the other two candidates are appended in order.
    assert result[0].wine.id == candidates[1].wine.id
    assert result[0].rerank_score == 0.8
    assert {r.wine.id for r in result} == {c.wine.id for c in candidates}


async def test_single_candidate_short_circuits() -> None:
    reranker = _reranker_returning(RuntimeError("should not be called"))
    candidates = _candidates(1)
    result = await reranker.rerank("q", candidates, top_k=5)
    assert result == candidates


async def test_noop_reranker_truncates_only() -> None:
    candidates = _candidates(5)
    result = await NoOpReranker().rerank("q", candidates, top_k=2)
    assert [r.wine.id for r in result] == [c.wine.id for c in candidates[:2]]
    assert all(r.rerank_score is None for r in result)
