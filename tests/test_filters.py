"""Tests for the WineFilters -> Qdrant Filter translation (I-5)."""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter

from hedonism_assistant.models.query import PriceRange, VintageRange, WineFilters
from hedonism_assistant.models.wine import WineCategory, WineColor
from hedonism_assistant.vector_store.filters import build_qdrant_filter
from hedonism_assistant.vector_store.payload import MAX_CRITIC_SCORE_FIELD


def _conditions(f: Filter | None) -> dict[str, FieldCondition]:
    """Index a filter's ``must`` conditions by their payload key."""
    assert f is not None
    return {c.key: c for c in f.must}  # type: ignore[union-attr]


def test_empty_filters_yield_no_filter() -> None:
    assert build_qdrant_filter(WineFilters()) is None


def test_single_color_is_match_any() -> None:
    conds = _conditions(build_qdrant_filter(WineFilters(color=[WineColor.RED])))
    assert conds["color"].match.any == ["red"]


def test_producer_is_match_any() -> None:
    # A producer query ("Dom Pérignon") becomes an exact OR-match filter.
    conds = _conditions(build_qdrant_filter(WineFilters(producer=["Dom Perignon"])))
    assert conds["producer"].match.any == ["Dom Perignon"]


def test_red_bordeaux_under_50_filters_all_three_dimensions() -> None:
    # The acceptance query: colour, region and price must all become hard filters.
    filters = WineFilters(
        color=[WineColor.RED],
        region=["Bordeaux"],
        price_range=PriceRange(max=50),
    )
    conds = _conditions(build_qdrant_filter(filters))

    assert conds["color"].match.any == ["red"]
    assert conds["region"].match.any == ["Bordeaux"]
    assert conds["price"].range.lte == 50
    assert conds["price"].range.gte is None


def test_min_critic_score_maps_to_unified_field() -> None:
    conds = _conditions(build_qdrant_filter(WineFilters(min_critic_score=90)))
    assert MAX_CRITIC_SCORE_FIELD in conds
    assert conds[MAX_CRITIC_SCORE_FIELD].range.gte == 90


def test_vintage_range_uses_only_present_bounds() -> None:
    conds = _conditions(build_qdrant_filter(WineFilters(vintage_range=VintageRange(min=2015))))
    assert conds["vintage"].range.gte == 2015
    assert conds["vintage"].range.lte is None

    conds = _conditions(build_qdrant_filter(WineFilters(vintage_range=VintageRange(max=2010))))
    assert conds["vintage"].range.lte == 2010
    assert conds["vintage"].range.gte is None


def test_bottle_size_and_in_bond_are_match_value() -> None:
    conds = _conditions(build_qdrant_filter(WineFilters(bottle_size_ml=1500, in_bond=True)))
    assert conds["bottle_size_ml"].match.value == 1500
    assert conds["in_bond"].match.value is True


def test_enum_category_serialises_to_string_value() -> None:
    conds = _conditions(build_qdrant_filter(WineFilters(category=[WineCategory.SPARKLING])))
    assert conds["category"].match.any == ["sparkling"]


def test_empty_range_bounds_drop_the_condition() -> None:
    # A range with neither bound carries no constraint.
    assert build_qdrant_filter(WineFilters(vintage_range=VintageRange())) is None
