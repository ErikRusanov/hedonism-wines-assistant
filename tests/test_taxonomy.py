"""Unit tests for catalogue-taxonomy filter validation."""

from hedonism_assistant.retrieval.taxonomy import Taxonomy, TaxonomyDimension
from tests.fixtures.wines import make_wine


def test_from_wines_collects_distinct_values() -> None:
    wines = [
        make_wine(
            producer="Pichon Lalande",
            country="France",
            region="Bordeaux",
            sub_region="Pauillac",
            grapes=["Merlot"],
        ),
        make_wine(
            producer="Domaine Leflaive",
            country="France",
            region="Burgundy",
            grapes=["Pinot Noir", "Merlot"],
        ),
        make_wine(country="Italy", region="Tuscany", grapes=[]),
    ]

    taxonomy = Taxonomy.from_wines(wines)

    assert taxonomy.producers == frozenset({"Pichon Lalande", "Domaine Leflaive"})
    assert taxonomy.countries == frozenset({"France", "Italy"})
    assert taxonomy.regions == frozenset({"Bordeaux", "Burgundy", "Tuscany"})
    assert taxonomy.sub_regions == frozenset({"Pauillac"})
    assert taxonomy.grapes == frozenset({"Merlot", "Pinot Noir"})


def test_canonicalize_is_diacritic_insensitive() -> None:
    # The catalogue stores "Dom Perignon" (no accent); a user/LLM writes the
    # accented form. Folding lets them match and returns the canonical spelling.
    taxonomy = Taxonomy(producers=frozenset({"Dom Perignon"}))

    assert taxonomy.canonicalize(TaxonomyDimension.PRODUCER, ["Dom Pérignon"]) == ["Dom Perignon"]


def test_canonicalize_drops_unknown_producer() -> None:
    # A hallucinated producer is dropped, degrading to pure semantic search
    # rather than silently zeroing out retrieval with a bogus hard filter.
    taxonomy = Taxonomy(producers=frozenset({"Dom Perignon"}))

    assert taxonomy.canonicalize(TaxonomyDimension.PRODUCER, ["Chateau Imaginaire"]) == []


def test_canonicalize_is_case_insensitive() -> None:
    taxonomy = Taxonomy(regions=frozenset({"Bordeaux", "Burgundy"}))

    assert taxonomy.canonicalize(TaxonomyDimension.REGION, ["bordeaux", "BURGUNDY"]) == [
        "Bordeaux",
        "Burgundy",
    ]


def test_canonicalize_drops_unknown_values() -> None:
    taxonomy = Taxonomy(regions=frozenset({"Bordeaux"}))

    assert taxonomy.canonicalize(TaxonomyDimension.REGION, ["Bordeaux", "Atlantis"]) == ["Bordeaux"]


def test_canonicalize_passes_through_on_empty_dimension() -> None:
    taxonomy = Taxonomy()

    assert taxonomy.canonicalize(TaxonomyDimension.REGION, ["  Bordeaux  ", "Rioja"]) == [
        "Bordeaux",
        "Rioja",
    ]


def test_canonicalize_dedupes_and_preserves_order() -> None:
    taxonomy = Taxonomy(grapes=frozenset({"Merlot", "Syrah"}))

    assert taxonomy.canonicalize(
        TaxonomyDimension.GRAPE, ["Syrah", "merlot", "SYRAH", "", "  "]
    ) == ["Syrah", "Merlot"]
