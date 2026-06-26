"""Tests for golden-case loading and predicate-based relevance (I-8)."""

from __future__ import annotations

from pathlib import Path

from hedonism_assistant.eval.golden import GoldenCase, load_golden, matches, relevant_ids
from hedonism_assistant.models.query import (
    PriceRange,
    QueryIntent,
    VintageRange,
    WineFilters,
)
from hedonism_assistant.models.wine import WineColor
from tests.fixtures.wines import sample_wines

# Reference cards: HED1001 red Bordeaux/Pauillac £320 (Vinous 96); HED1002 white
# Burgundy/Chablis £45; HED1003 Italian red £210; HED1005 in-bond Bordeaux £9500.


def _by_id(wine_id: str):
    return next(w for w in sample_wines() if w.id == wine_id)


def test_matches_color_and_region() -> None:
    spec = WineFilters(color=[WineColor.RED], region=["Bordeaux"])
    assert matches(_by_id("HED1001"), spec)  # red Bordeaux
    assert not matches(_by_id("HED1002"), spec)  # white Burgundy


def test_matches_is_case_insensitive_on_strings() -> None:
    assert matches(_by_id("HED1001"), WineFilters(region=["bordeaux"]))
    assert matches(_by_id("HED1003"), WineFilters(country=["ITALY"]))


def test_matches_price_ceiling() -> None:
    assert matches(_by_id("HED1002"), WineFilters(price_range=PriceRange(max=50)))
    assert not matches(_by_id("HED1001"), WineFilters(price_range=PriceRange(max=50)))


def test_matches_grapes_intersection() -> None:
    assert matches(_by_id("HED1002"), WineFilters(grapes=["Chardonnay"]))
    assert not matches(_by_id("HED1002"), WineFilters(grapes=["Merlot"]))


def test_matches_vintage_range_excludes_nv() -> None:
    spec = WineFilters(vintage_range=VintageRange(min=2014, max=2016))
    assert matches(_by_id("HED1001"), spec)  # 2015
    assert not matches(_by_id("HED1004"), spec)  # NV Champagne, no vintage


def test_matches_min_critic_score_normalises_scales() -> None:
    # HED1003 has a 20-pt Jancis 18 (= 90/100) and a 100-pt Vinous 95; max is 95.
    assert matches(_by_id("HED1003"), WineFilters(min_critic_score=95))
    assert not matches(_by_id("HED1003"), WineFilters(min_critic_score=96))


def test_matches_in_bond_flag() -> None:
    assert matches(_by_id("HED1005"), WineFilters(in_bond=True))
    assert not matches(_by_id("HED1001"), WineFilters(in_bond=True))


def test_empty_spec_matches_everything() -> None:
    assert all(matches(w, WineFilters()) for w in sample_wines())


def test_relevant_ids_predicate() -> None:
    case = GoldenCase(
        id="G1",
        question="red Bordeaux",
        relevance=WineFilters(color=[WineColor.RED], region=["Bordeaux"]),
    )
    ids = relevant_ids(case, sample_wines())
    # Red Bordeaux only: Pichon Lalande and Latour. d'Yquem (HED1006) is a white
    # Sauternes — same region, wrong colour — so the colour gate excludes it.
    assert ids == {"HED1001", "HED1005"}


def test_relevant_ids_id_pin_overrides_predicate() -> None:
    case = GoldenCase(
        id="G2",
        question="that Latour",
        expected_wine_ids=["HED1005"],
        relevance=WineFilters(color=[WineColor.WHITE]),  # ignored when ids are pinned
    )
    assert relevant_ids(case, sample_wines()) == {"HED1005"}


def test_load_golden_roundtrips_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    cases = [
        GoldenCase(
            id="G1",
            question="red Bordeaux under £100",
            expected_intent=QueryIntent.RECOMMENDATION,
            relevance=WineFilters(color=[WineColor.RED], region=["Bordeaux"]),
        ),
        GoldenCase(id="G2", question="any whisky?", expected_fallback="other_drinks"),
    ]
    path.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")

    loaded = load_golden(path)

    assert [c.id for c in loaded] == ["G1", "G2"]
    assert loaded[1].expected_fallback == "other_drinks"


def test_load_golden_skips_blank_and_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    good = GoldenCase(id="G1", question="q").model_dump_json()
    path.write_text(f"{good}\n\n   \n{{not json}}\n", encoding="utf-8")

    loaded = load_golden(path)

    assert [c.id for c in loaded] == ["G1"]
