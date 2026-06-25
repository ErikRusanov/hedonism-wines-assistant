"""Unit tests for the deterministic guardrail fallbacks (I-6)."""

from __future__ import annotations

from hedonism_assistant.generation.fallbacks import (
    empty_retrieval_suggestions,
    out_of_scope_suggestions,
)
from hedonism_assistant.models.query import PriceRange, VintageRange, WineFilters
from hedonism_assistant.models.wine import WineColor


def test_out_of_scope_suggestions_capped() -> None:
    assert len(out_of_scope_suggestions(limit=2)) == 2
    assert out_of_scope_suggestions(limit=0) == []
    # Asking for more than exist just returns all of them.
    assert len(out_of_scope_suggestions(limit=99)) >= 1


def test_empty_retrieval_suggests_relaxing_price_first() -> None:
    filters = WineFilters(price_range=PriceRange(max=50), region=["Bordeaux"])

    suggestions = empty_retrieval_suggestions(filters, limit=3)

    assert "£50" in suggestions[0]
    assert any("Bordeaux" in s for s in suggestions)


def test_empty_retrieval_covers_each_filter_kind() -> None:
    filters = WineFilters(
        color=[WineColor.RED],
        grapes=["Nebbiolo"],
        vintage_range=VintageRange(min=2015, max=2015),
    )

    suggestions = empty_retrieval_suggestions(filters, limit=10)

    assert any("Nebbiolo" in s for s in suggestions)
    assert any("vintage" in s.lower() for s in suggestions)
    assert any("colour" in s.lower() for s in suggestions)


def test_empty_retrieval_capped_at_limit() -> None:
    filters = WineFilters(
        price_range=PriceRange(min=10, max=50),
        region=["Bordeaux"],
        country=["France"],
        grapes=["Merlot"],
    )
    assert len(empty_retrieval_suggestions(filters, limit=2)) == 2


def test_empty_retrieval_generic_when_no_filters() -> None:
    suggestions = empty_retrieval_suggestions(WineFilters(), limit=3)
    assert len(suggestions) == 1
    assert "broader" in suggestions[0].lower() or "differently" in suggestions[0].lower()
