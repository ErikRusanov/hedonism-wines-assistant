"""Tests for parsing a product page into a RawWine."""

from pathlib import Path

import pytest

from hedonism_assistant.data.models import RawWine
from hedonism_assistant.data.parser import parse_product, product_markup_missing
from hedonism_assistant.models.wine import Availability, WineColor

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def wine() -> RawWine:
    html = (FIXTURES / "product_wine.html").read_text(encoding="utf-8")
    parsed = parse_product(html, "https://hedonism.co.uk/product/chateau-imaginaire-2018")
    assert parsed is not None
    return parsed


def test_jsonld_fields(wine: RawWine) -> None:
    assert wine.sku == "HED99999"
    assert wine.name == "Chateau Imaginaire 2018"
    assert wine.price == 55.0
    assert wine.currency == "GBP"
    assert wine.availability is Availability.IN_STOCK
    assert wine.image_url is not None and wine.image_url.endswith("HED99999.JPG")
    assert wine.tasting_notes is not None and "Dark fruit" in wine.tasting_notes


def test_html_spec_fields(wine: RawWine) -> None:
    assert wine.slug == "chateau-imaginaire-2018"
    assert wine.section == "Wines"
    assert wine.is_wine is True
    assert wine.color is WineColor.RED
    assert wine.vintage == 2018
    assert wine.bottle_size_ml == 750
    assert wine.size_raw == "75cl"
    assert wine.abv == 13.5
    assert wine.price_ex_vat == 45.83
    assert wine.stock_qty == 4


def test_attributes_are_scoped_to_the_product(wine: RawWine) -> None:
    # The "you may also like" rail lists a different producer/grape; the parser
    # must read the product's own attributes, not the rail's.
    assert wine.producer == "Imaginaire Estate"
    assert wine.grapes == ["Cabernet Sauvignon"]
    assert wine.region == "Bordeaux"
    assert wine.country == "France"


def test_critic_scores_carry_the_right_scale(wine: RawWine) -> None:
    by_critic = {c.critic: c for c in wine.critic_scores}
    assert by_critic["Robert Parker"].score == 95.0
    assert by_critic["Robert Parker"].scale == 100
    assert by_critic["Jancis Robinson"].score == 18.0
    assert by_critic["Jancis Robinson"].scale == 20


def test_spirit_is_recognised_as_non_wine() -> None:
    html = (FIXTURES / "product_spirit.html").read_text(encoding="utf-8")
    raw = parse_product(html, "https://hedonism.co.uk/product/invented-single-malt-whisky")
    assert raw is not None
    assert raw.section == "Spirits"
    assert raw.is_wine is False
    assert raw.availability is Availability.OUT_OF_STOCK
    assert raw.stock_qty is None  # "Out of stock" carries no count


def test_non_vintage_and_sale_pricing() -> None:
    html = """
    <ol class="breadcrumb">
      <li><a href="/">Home</a></li><li><a href="/wines">Wines</a></li>
    </ol>
    <div class="product_intro">
      <h1 class="page-title"><span>Invented Champagne NV</span></h1>
      <div class="title-sub-string"><span>HED11111 </span><span>150cl</span><span> 12% </span></div>
      <div class="field--name-price">
        <div class="price-with-discount">
          <del class="was-price">£100.00</del>
          <div class="base-price">£80.00</div>
        </div>
      </div>
      <div class="attribute-item vintage"><span class="field--item">NV</span></div>
      <div class="attribute-item colour">White</div>
    </div>"""
    raw = parse_product(html, "https://hedonism.co.uk/product/invented-champagne-nv")
    assert raw is not None
    assert raw.vintage is None
    assert raw.bottle_size_ml == 1500
    assert raw.price == 80.0
    assert raw.on_sale is True
    assert raw.sale_was_price == 100.0
    assert raw.color is WineColor.WHITE


def test_non_product_page_returns_none() -> None:
    assert parse_product("<html><body><h1>Hello</h1></body></html>", "https://x/y") is None


def test_product_markup_missing_predicate() -> None:
    assert product_markup_missing("<html>no markup here</html>") is True
    assert product_markup_missing('<div class="product_intro"></div>') is False
