"""Unit tests for marker-based citation extraction (I-6).

The function is pure — no model, no network — so these tests pin the parsing and
ordering rules directly.
"""

from __future__ import annotations

from hedonism_assistant.generation.citations import extract_citations
from hedonism_assistant.models.wine import RetrievedWine
from tests.fixtures.wines import sample_wines


def _retrieved() -> list[RetrievedWine]:
    return [RetrievedWine(wine=w, score=1.0 - i * 0.1) for i, w in enumerate(sample_wines())]


def test_maps_markers_in_first_mention_order() -> None:
    retrieved = _retrieved()
    answer = "I'd pick the Brunello [3], then the Chablis [2] for something lighter."

    citations = extract_citations(answer, retrieved)

    assert [c.wine_id for c in citations] == [retrieved[2].wine.id, retrieved[1].wine.id]
    assert citations[0].name == retrieved[2].wine.name
    assert citations[0].price == retrieved[2].wine.price


def test_deduplicates_repeated_markers() -> None:
    retrieved = _retrieved()
    answer = "The Pichon [1] is superb; really, the Pichon [1] stands out [1]."

    citations = extract_citations(answer, retrieved)

    assert [c.wine_id for c in citations] == [retrieved[0].wine.id]


def test_drops_out_of_range_and_zero_markers() -> None:
    retrieved = _retrieved()[:3]
    answer = "Maybe [99] or [0], but the Chablis [2] is the safe choice."

    citations = extract_citations(answer, retrieved)

    assert [c.wine_id for c in citations] == [retrieved[1].wine.id]


def test_no_markers_yields_no_citations() -> None:
    answer = "Here are some lovely wines, but I won't number them."
    assert extract_citations(answer, _retrieved()) == []


def test_empty_retrieved_yields_no_citations() -> None:
    assert extract_citations("the wine [1] is great", []) == []
