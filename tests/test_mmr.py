"""Unit tests for MMR diversification (I-5)."""

from __future__ import annotations

from hedonism_assistant.models.wine import RetrievedWine
from hedonism_assistant.retrieval.mmr import mmr_select
from tests.fixtures.wines import make_wine


def _candidate(wine_id: str, score: float) -> RetrievedWine:
    return RetrievedWine(wine=make_wine(id=wine_id), score=score)


def test_mmr_prefers_diversity_over_a_near_duplicate() -> None:
    # a and b are near-identical; c is orthogonal. a is most relevant, b second.
    a = (_candidate("A", 1.0), [1.0, 0.0])
    b = (_candidate("B", 0.9), [0.99, 0.01])
    c = (_candidate("C", 0.5), [0.0, 1.0])

    # Diversity-leaning lambda: after picking a, the orthogonal c beats the
    # near-duplicate b despite b's higher relevance.
    result = mmr_select([a, b, c], lambda_=0.3, top_k=2)
    assert [r.wine.id for r in result] == ["A", "C"]


def test_mmr_pure_relevance_keeps_score_order() -> None:
    a = (_candidate("A", 1.0), [1.0, 0.0])
    b = (_candidate("B", 0.9), [0.99, 0.01])
    c = (_candidate("C", 0.5), [0.0, 1.0])

    # lambda=1 ignores diversity entirely -> pure relevance order.
    result = mmr_select([a, b, c], lambda_=1.0, top_k=3)
    assert [r.wine.id for r in result] == ["A", "B", "C"]


def test_mmr_truncates_to_top_k() -> None:
    pairs = [(_candidate(f"W{i}", 1.0 - i * 0.1), [float(i), 0.0]) for i in range(5)]
    assert len(mmr_select(pairs, lambda_=0.5, top_k=3)) == 3
