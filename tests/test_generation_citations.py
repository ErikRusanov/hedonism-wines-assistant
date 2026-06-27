"""Unit tests for marker-based citation extraction (I-6).

The function is pure — no model, no network — so these tests pin the parsing and
ordering rules directly.
"""

from __future__ import annotations

from hedonism_assistant.generation.citations import extract_citations
from hedonism_assistant.models.wine import CriticScore, RetrievedWine
from tests.fixtures.wines import make_wine, sample_wines


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


def test_citation_carries_card_fields_for_rendering() -> None:
    retrieved = _retrieved()
    wine = retrieved[0].wine  # Pichon Lalande 2015, Vinous 96/100

    [citation] = extract_citations("the Pichon [1]", retrieved)

    assert citation.producer == wine.producer
    assert citation.region == wine.region
    assert citation.vintage == wine.vintage
    assert citation.color == wine.color
    assert citation.grapes == wine.grapes
    # Image is served first-party from the SKU, never the catalogue CDN URL.
    assert citation.image_path == f"/bottles/{wine.id}.jpg"
    assert citation.top_critic == "Vinous"
    assert citation.top_critic_score == 96.0


def test_citation_top_critic_is_none_without_scores() -> None:
    retrieved = _retrieved()
    wine = retrieved[1].wine  # Chablis Droin 2022, no critic scores

    [citation] = extract_citations("the Chablis [2]", retrieved)

    assert citation.top_critic is None
    assert citation.top_critic_score is None
    assert citation.image_path == f"/bottles/{wine.id}.jpg"


def test_citation_skips_critic_scores_that_normalise_above_100() -> None:
    # A 100-point value mislabelled as a 20-point scale (92/20 -> 460) is an
    # extraction error; the valid Vinous 93 should win, never the bogus one.
    wine = make_wine(
        id="HED9",
        critic_scores=[
            CriticScore(critic="Jancis Robinson", score=92.0, scale=20),
            CriticScore(critic="Vinous", score=93.0, scale=100),
        ],
    )
    retrieved = [RetrievedWine(wine=wine, score=1.0)]

    [citation] = extract_citations("the wine [1]", retrieved)

    assert citation.top_critic == "Vinous"
    assert citation.top_critic_score == 93.0


def test_citation_with_only_bad_scores_has_no_critic() -> None:
    wine = make_wine(
        id="HED10",
        critic_scores=[CriticScore(critic="Jancis Robinson", score=92.0, scale=20)],
    )
    retrieved = [RetrievedWine(wine=wine, score=1.0)]

    [citation] = extract_citations("the wine [1]", retrieved)

    assert citation.top_critic is None
    assert citation.top_critic_score is None
