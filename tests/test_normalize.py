"""Tests for the deterministic RawWine -> Wine normalization (I-2)."""

from hedonism_assistant.data.models import RawWine
from hedonism_assistant.data.normalize import (
    build_embedding_text,
    canonicalize_country,
    canonicalize_grapes,
    classify_category,
    format_name,
    normalize_wine,
)
from hedonism_assistant.models.wine import CriticScore, Wine, WineCategory, WineColor


def _raw(**overrides: object) -> RawWine:
    base: dict[str, object] = {
        "url": "https://hedonism.co.uk/product/x",
        "slug": "x",
        "sku": "HED1",
        "name": "Test Wine",
        "section": "Wines",
        "bottle_size_ml": 750,
        "price": 50.0,
    }
    base.update(overrides)
    return RawWine(**base)


# --- Category classification --------------------------------------------------
def test_category_defaults_to_still() -> None:
    assert classify_category(_raw(name="Château Test 2018", region="Bordeaux")) is (
        WineCategory.STILL
    )


def test_category_sparkling_from_champagne() -> None:
    assert classify_category(_raw(name="Bollinger Special Cuvée NV", region="Champagne")) is (
        WineCategory.SPARKLING
    )


def test_category_sweet_from_region() -> None:
    assert classify_category(_raw(name="Yquem 2015", region="Sauternes")) is WineCategory.SWEET


def test_category_fortified_from_port() -> None:
    assert classify_category(_raw(name="Taylor's Vintage Port 2017", region="Douro")) is (
        WineCategory.FORTIFIED
    )


def test_fortified_wins_over_sweet() -> None:
    # Pedro Ximénez sherry is intensely sweet but is a fortified wine.
    assert classify_category(_raw(name="Pedro Ximénez Sherry")) is WineCategory.FORTIFIED


def test_sparkling_wins_over_sweet() -> None:
    # Moscato d'Asti is sweet but classed as sparkling.
    assert classify_category(_raw(name="Moscato d'Asti 2022")) is WineCategory.SPARKLING


def test_port_in_country_does_not_falsely_match() -> None:
    # "Portugal" must not trigger the 'port' fortified keyword.
    assert classify_category(_raw(name="Quinta Dry Red 2019", region="Portugal")) is (
        WineCategory.STILL
    )


# --- Grape & country canonicalisation ----------------------------------------
def test_grapes_canonicalised_and_deduped() -> None:
    assert canonicalize_grapes(["Tinto Fino", "Shiraz", "tinto fino", " "]) == [
        "Tempranillo",
        "Syrah",
    ]


def test_country_canonicalised() -> None:
    assert canonicalize_country("United States") == "USA"
    assert canonicalize_country("Spain") == "Spain"
    assert canonicalize_country(None) is None


# --- Bottle format ------------------------------------------------------------
def test_format_name_table() -> None:
    assert format_name(750) is None
    assert format_name(1500) == "Magnum"
    assert format_name(3000) == "Double Magnum"
    assert format_name(375) == "Half"
    assert format_name(900) is None


# --- normalize_wine -----------------------------------------------------------
def test_normalize_maps_and_canonicalises() -> None:
    wine = normalize_wine(
        _raw(grapes=["Tinto Fino"], country="United States", bottle_size_ml=1500, vintage=2019)
    )
    assert wine is not None
    assert wine.id == "HED1"
    assert wine.grapes == ["Tempranillo"]
    assert wine.country == "USA"
    assert wine.format_name == "Magnum"
    assert wine.category is WineCategory.STILL
    assert wine.embedding_text is None  # set later by the orchestrator


def test_normalize_drops_incomplete_record() -> None:
    assert normalize_wine(_raw(price=None)) is None


def test_normalize_drops_non_wine() -> None:
    assert normalize_wine(_raw(section="Spirits")) is None


# --- Embedding passport -------------------------------------------------------
def _wine(**overrides: object) -> Wine:
    base: dict[str, object] = {
        "id": "HED1",
        "slug": "x",
        "name": "Vega Sicilia Valbuena 2019",
        "url": "https://hedonism.co.uk/product/x",
        "category": WineCategory.STILL,
        "color": WineColor.RED,
        "producer": "Vega Sicilia",
        "country": "Spain",
        "region": "Ribera del Duero",
        "vintage": 2019,
        "grapes": ["Tempranillo"],
        "abv": 14.5,
        "bottle_size_ml": 3000,
        "format_name": "Double Magnum",
        "price": 1100.0,
        "currency": "GBP",
        "critic_scores": [CriticScore(critic="Vinous", score=94, scale=100)],
    }
    base.update(overrides)
    return Wine(**base)


def test_embedding_text_carries_structured_facts() -> None:
    text = build_embedding_text(_wine(), notes_chars=0)
    assert "Vega Sicilia Valbuena 2019 is a 2019 red still wine" in text
    assert "from Ribera del Duero, Spain" in text
    assert "produced by Vega Sicilia" in text
    assert "Tempranillo" in text
    assert "£1,100" in text
    assert "Double Magnum" in text
    assert "94/100 (Vinous)" in text


def test_embedding_text_includes_enrichment_tags() -> None:
    text = build_embedding_text(
        _wine(style_tags=["full-bodied", "oaked"], food_pairings=["roast lamb"]),
        notes_chars=0,
    )
    assert "Style: full-bodied and oaked." in text
    assert "Pairs with roast lamb." in text


def test_embedding_text_truncates_notes() -> None:
    notes = "First sentence here. " + "padding word " * 200
    text = build_embedding_text(_wine(tasting_notes=notes), notes_chars=80)
    # The whole note is far longer than the budget; the passport must stay bounded.
    assert len(text) < len(notes)
    assert text.endswith(".") or text.endswith("…")
