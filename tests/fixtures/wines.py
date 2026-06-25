"""Canonical :class:`Wine` cards and a builder shared across the test suite.

``make_wine(**overrides)`` is the single source of a minimally-valid card (it
replaces the per-file ``_wine`` helpers). ``sample_wines()`` returns a small,
deliberately diverse set — red Bordeaux, white Burgundy, Italian still,
Champagne, an in-bond bottle, and one with mixed critic scales — enough to
exercise the payload builder, critic-score normalisation and indexing.
"""

from __future__ import annotations

from hedonism_assistant.models.wine import (
    CriticScore,
    Wine,
    WineCategory,
    WineColor,
)


def make_wine(**overrides: object) -> Wine:
    """Build a minimally-valid :class:`Wine`, overriding any field by keyword."""
    base: dict[str, object] = {
        "id": "HED1",
        "slug": "a-wine",
        "name": "A Wine",
        "url": "https://hedonism.co.uk/product/a-wine",
        "category": WineCategory.STILL,
        "bottle_size_ml": 750,
        "price": 42.0,
    }
    base.update(overrides)
    return Wine(**base)


def sample_wines() -> list[Wine]:
    """A diverse handful of cards covering the payload/index code paths."""
    return [
        make_wine(
            id="HED1001",
            slug="pichon-lalande-2015",
            name="Pichon Lalande 2015",
            url="https://hedonism.co.uk/product/pichon-lalande-2015",
            color=WineColor.RED,
            producer="Pichon Lalande",
            country="France",
            region="Bordeaux",
            sub_region="Pauillac",
            classification="Second Growth",
            vintage=2015,
            grapes=["Cabernet Sauvignon", "Merlot"],
            price=320.0,
            critic_scores=[CriticScore(critic="Vinous", score=96, scale=100)],
            embedding_text="Pichon Lalande 2015 is a red Bordeaux from Pauillac.",
        ),
        make_wine(
            id="HED1002",
            slug="chablis-droin-2022",
            name="Chablis Droin 2022",
            url="https://hedonism.co.uk/product/chablis-droin-2022",
            color=WineColor.WHITE,
            producer="Jean-Paul Droin",
            country="France",
            region="Burgundy",
            sub_region="Chablis",
            vintage=2022,
            grapes=["Chardonnay"],
            price=45.0,
            embedding_text="Chablis Droin 2022 is a white Burgundy from Chablis.",
        ),
        make_wine(
            id="HED1003",
            slug="brunello-biondi-santi-2016",
            name="Brunello Biondi-Santi 2016",
            url="https://hedonism.co.uk/product/brunello-biondi-santi-2016",
            color=WineColor.RED,
            producer="Biondi-Santi",
            country="Italy",
            region="Tuscany",
            sub_region="Montalcino",
            vintage=2016,
            grapes=["Sangiovese"],
            price=210.0,
            # Mixed scales: 100-pt Vinous and 20-pt Jancis Robinson.
            critic_scores=[
                CriticScore(critic="Vinous", score=95, scale=100),
                CriticScore(critic="Jancis Robinson", score=18, scale=20),
            ],
            embedding_text="Brunello Biondi-Santi 2016 is an Italian Sangiovese from Montalcino.",
        ),
        make_wine(
            id="HED1004",
            slug="dom-perignon-nv",
            name="Dom Perignon NV",
            url="https://hedonism.co.uk/product/dom-perignon-nv",
            category=WineCategory.SPARKLING,
            color=WineColor.WHITE,
            producer="Moet & Chandon",
            country="France",
            region="Champagne",
            vintage=None,
            grapes=["Chardonnay", "Pinot Noir"],
            price=180.0,
            embedding_text="Dom Perignon NV is a Champagne sparkling wine.",
        ),
        make_wine(
            id="HED1005",
            slug="latour-2010-in-bond",
            name="Chateau Latour 2010",
            url="https://hedonism.co.uk/product/latour-2010-in-bond",
            color=WineColor.RED,
            producer="Chateau Latour",
            country="France",
            region="Bordeaux",
            sub_region="Pauillac",
            classification="First Growth",
            vintage=2010,
            grapes=["Cabernet Sauvignon"],
            price=9500.0,
            in_bond=True,
            critic_scores=[CriticScore(critic="Parker", score=100, scale=100)],
            embedding_text="Chateau Latour 2010 is an in-bond first-growth Bordeaux.",
        ),
        make_wine(
            id="HED1006",
            slug="sauternes-yquem-2017",
            name="Chateau d'Yquem 2017",
            url="https://hedonism.co.uk/product/sauternes-yquem-2017",
            category=WineCategory.SWEET,
            color=WineColor.WHITE,
            producer="Chateau d'Yquem",
            country="France",
            region="Bordeaux",
            sub_region="Sauternes",
            vintage=2017,
            grapes=["Semillon", "Sauvignon Blanc"],
            price=420.0,
            embedding_text="Chateau d'Yquem 2017 is a sweet Sauternes dessert wine.",
        ),
    ]
