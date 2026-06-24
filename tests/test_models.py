"""Contract tests pinning the shape of the core domain models."""

from hedonism_assistant.models import (
    Availability,
    CriticScore,
    ParsedQuery,
    QueryIntent,
    RetrievedWine,
    Wine,
    WineCategory,
    WineColor,
)


def _sample_wine() -> Wine:
    return Wine(
        id="HED12345",
        slug="chateau-margaux-2015",
        name="Chateau Margaux 2015",
        url="https://hedonism.co.uk/product/chateau-margaux-2015",
        category=WineCategory.STILL,
        color=WineColor.RED,
        producer="Chateau Margaux",
        region="Bordeaux",
        sub_region="Margaux",
        country="France",
        vintage=2015,
        grapes=["Cabernet Sauvignon", "Merlot"],
        bottle_size_ml=750,
        price=950.0,
        critic_scores=[CriticScore(critic="Vinous", score=98)],
    )


def test_wine_defaults() -> None:
    wine = _sample_wine()
    assert wine.currency == "GBP"
    assert wine.grapes == ["Cabernet Sauvignon", "Merlot"]
    assert wine.embedding_text is None
    assert wine.availability is Availability.IN_STOCK
    assert wine.in_bond is False
    assert wine.critic_scores[0].scale == 100


def test_retrieved_wine_carries_scores() -> None:
    retrieved = RetrievedWine(wine=_sample_wine(), score=0.87, rerank_score=0.95)
    assert retrieved.score == 0.87
    assert retrieved.rerank_score == 0.95


def test_parsed_query_defaults() -> None:
    parsed = ParsedQuery(semantic_query="red bordeaux under 50")
    assert parsed.intent == QueryIntent.RECOMMENDATION
    assert parsed.confident is True
    assert parsed.filters.color == []
