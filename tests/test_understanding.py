"""Unit tests for the query-understanding chip projection.

``filters_to_chips`` is pure, so these pin the label shapes the UI shows as
read-only "Understood" pills.
"""

from __future__ import annotations

from hedonism_assistant.generation.understanding import filters_to_chips
from hedonism_assistant.models.query import ParsedQuery, PriceRange, VintageRange, WineFilters
from hedonism_assistant.models.wine import WineCategory, WineColor


def _parsed(**filter_kwargs) -> ParsedQuery:
    return ParsedQuery(semantic_query="q", filters=WineFilters(**filter_kwargs))


def test_color_region_and_price_cap() -> None:
    parsed = _parsed(color=[WineColor.RED], region=["Bordeaux"], price_range=PriceRange(max=50))
    assert filters_to_chips(parsed) == ["Red", "Bordeaux", "under £50"]


def test_price_band_and_floor() -> None:
    assert filters_to_chips(_parsed(price_range=PriceRange(min=30, max=50))) == ["£30–£50"]
    assert filters_to_chips(_parsed(price_range=PriceRange(min=100))) == ["over £100"]


def test_vintage_single_year_and_range() -> None:
    single = filters_to_chips(_parsed(vintage_range=VintageRange(min=2015, max=2015)))
    band = filters_to_chips(_parsed(vintage_range=VintageRange(min=2015, max=2018)))
    assert single == ["2015"]
    assert band == ["2015–2018"]


def test_category_grape_and_critic_score() -> None:
    parsed = _parsed(category=[WineCategory.SPARKLING], grapes=["Nebbiolo"], min_critic_score=92)
    assert filters_to_chips(parsed) == ["Sparkling", "Nebbiolo", "92+ pts"]


def test_bottle_size_in_litres() -> None:
    assert filters_to_chips(_parsed(bottle_size_ml=1500)) == ["1.5L"]


def test_no_filters_yields_no_chips() -> None:
    assert filters_to_chips(ParsedQuery(semantic_query="something nice")) == []


def test_chips_deduplicate_case_insensitively() -> None:
    # A producer can echo a region; the chip list should not repeat it.
    parsed = _parsed(region=["Champagne"], producer=["champagne"])
    assert filters_to_chips(parsed) == ["Champagne"]
