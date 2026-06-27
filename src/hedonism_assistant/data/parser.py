"""Parse a product page into a :class:`RawWine`.

Strategy mirrors the page itself: structured ``schema.org/Product`` JSON-LD is the
source of truth for the fields it carries (name, SKU, price, currency,
availability, image, description), and the rendered HTML spec block supplies the
rest. Crucially, the HTML reads are *scoped to* the ``.product_intro`` container:
"you may also like" and "other vintages" rails repeat ``/wineries/`` and
``/wine/`` links for unrelated bottles, so an unscoped lookup would pick up the
wrong producer or grape.

Everything is best-effort and defensive -- a missing or malformed field never
aborts the record. The breadcrumb section is captured so the orchestrator can
keep wines and drop spirits/accessories that share the catalogue listing.

The parsing logic lives on :class:`ProductParser`, which binds one page's
``soup``/``url`` once and exposes ``@classmethod`` entry points; the pure
field-level coercions are ``@staticmethod`` so they stay independently testable
and free of page state. :func:`parse_product` is kept as a thin module-level
facade for callers and tests.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import singledispatch
from typing import Any, Final
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag

from hedonism_assistant.data.models import RawWine
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.wine import Availability, CriticScore, WineColor

logger = get_logger(__name__)

_COLOR_BY_LABEL: Final[dict[str, WineColor]] = {
    "red": WineColor.RED,
    "white": WineColor.WHITE,
    "rose": WineColor.ROSE,
    "rosé": WineColor.ROSE,
}

# Critics rated on a 20-point scale; everyone else is treated as /100.
_SCALE_20_CRITICS: Final[frozenset[str]] = frozenset({"jancis robinson", "jr"})

_AVAILABILITY_BY_SCHEMA: Final[dict[str, Availability]] = {
    "instock": Availability.IN_STOCK,
    "limitedavailability": Availability.IN_STOCK,
    "onlineonly": Availability.IN_STOCK,
    "preorder": Availability.IN_STOCK,
    "backorder": Availability.IN_STOCK,
    "outofstock": Availability.OUT_OF_STOCK,
    "soldout": Availability.OUT_OF_STOCK,
    "discontinued": Availability.OUT_OF_STOCK,
}

# Precompiled once at import rather than per call: parsing tens of thousands of
# pages means these run in tight loops.
_RE_PRICE: Final[re.Pattern[str]] = re.compile(r"\d[\d,]*(?:\.\d+)?")
_RE_SIZE: Final[re.Pattern[str]] = re.compile(
    r"\s*([\d.]+)\s*(cl|ml|l|litre|liter)\s*", re.IGNORECASE
)
_RE_ABV: Final[re.Pattern[str]] = re.compile(r"\s*([\d.]+)\s*%\s*")
_RE_VINTAGE: Final[re.Pattern[str]] = re.compile(r"(?:19|20)\d{2}")
_RE_INT: Final[re.Pattern[str]] = re.compile(r"\d+")
_RE_SCORE: Final[re.Pattern[str]] = re.compile(r"\s*([\d.]+)\s*\+?\s+(.+?)\s*$")
_RE_SKU: Final[re.Pattern[str]] = re.compile(r"HED\w+", re.IGNORECASE)
_RE_WS: Final[re.Pattern[str]] = re.compile(r"\s+")


@singledispatch
def _image_url(image: Any) -> str | None:
    """Pull the first usable image URL out of a JSON-LD ``image`` value.

    ``image`` may be a bare URL, an ``ImageObject`` dict, or a list mixing the
    two; dispatch keeps each shape's handling isolated. Unknown shapes (and a
    missing ``None`` value) fall through to this default and yield ``None``.
    """
    return None


@_image_url.register
def _(image: str) -> str | None:
    return image or None


@_image_url.register
def _(image: dict) -> str | None:
    return ProductParser._clean(image.get("url"))


@_image_url.register
def _(image: list) -> str | None:
    return next((url for item in image if (url := _image_url(item))), None)


class ProductParser:
    """Turn one product page's HTML into a :class:`RawWine` (or ``None``).

    Construct from raw HTML, or go through the :meth:`parse` classmethod. Page
    state (``soup``, ``url``) lives on the instance; everything that only
    transforms a value is a ``@staticmethod`` so it can be reused and tested
    without a page.
    """

    __slots__ = ("_soup", "_url")

    def __init__(self, html: str, url: str) -> None:
        self._soup = BeautifulSoup(html, "lxml")
        self._url = url

    @classmethod
    def parse(cls, html: str, url: str) -> RawWine | None:
        """Parse one product page; return ``None`` if it is not a product page."""
        return cls(html, url)._build()

    def _build(self) -> RawWine | None:
        intro = self._soup.select_one(".product_intro")
        jsonld = self._extract_jsonld_product()
        if intro is None and jsonld is None:
            logger.debug("not_a_product_page", url=self._url)
            return None

        raw = RawWine(
            url=self._url,
            slug=self._slug_from_url(self._url),
            section=self._breadcrumb_section(),
            fetched_at=datetime.now(UTC).isoformat(),
        )
        if jsonld is not None:
            self._apply_jsonld(raw, jsonld)
        if intro is not None:
            self._apply_html(raw, intro)
        self._apply_diet_badges(raw)
        return raw

    # ----------------------------------------------------------------------- #
    # JSON-LD                                                                  #
    # ----------------------------------------------------------------------- #
    def _extract_jsonld_product(self) -> dict[str, Any] | None:
        for script in self._soup.find_all("script", type="application/ld+json"):
            raw = script.string or script.get_text()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                continue
            for node in self._iter_jsonld_nodes(data):
                node_type = node.get("@type")
                if node_type == "Product" or (
                    isinstance(node_type, list) and "Product" in node_type
                ):
                    return node
        return None

    @staticmethod
    def _iter_jsonld_nodes(data: Any) -> Iterator[dict[str, Any]]:
        if isinstance(data, list):
            for item in data:
                yield from ProductParser._iter_jsonld_nodes(item)
        elif isinstance(data, dict):
            graph = data.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    yield from ProductParser._iter_jsonld_nodes(item)
            else:
                yield data

    @classmethod
    def _apply_jsonld(cls, raw: RawWine, product: dict[str, Any]) -> None:
        raw.name = cls._clean(product.get("name")) or raw.name
        raw.sku = cls._clean(product.get("sku")) or raw.sku
        raw.tasting_notes = cls._collapse(product.get("description")) or raw.tasting_notes
        raw.image_url = _image_url(product.get("image")) or raw.image_url

        offers = product.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else None
        if isinstance(offers, dict):
            raw.price = cls._parse_price(offers.get("price"))
            raw.currency = cls._clean(offers.get("priceCurrency")) or raw.currency
            raw.availability = cls._map_availability(offers.get("availability"))

    # ----------------------------------------------------------------------- #
    # HTML spec block (scoped to .product_intro)                              #
    # ----------------------------------------------------------------------- #
    @classmethod
    def _apply_html(cls, raw: RawWine, intro: Tag) -> None:
        if not raw.name:
            raw.name = cls._text(intro.select_one("h1.page-title")) or None

        sub = intro.select_one(".title-sub-string")
        if sub is not None:
            cls._apply_sub_string(raw, sub)

        cls._apply_price_block(raw, intro)
        cls._apply_attributes(raw, intro)

        raw.stock_qty = cls._parse_int(cls._text(intro.select_one(".product-stock")))
        raw.critic_scores = cls._parse_scores(intro)
        raw.in_bond = cls._detect_in_bond(intro)

    @classmethod
    def _apply_sub_string(cls, raw: RawWine, sub: Tag) -> None:
        """Read the "HED27711 75cl 14%" line: SKU, bottle size and ABV."""
        for span in sub.find_all("span", recursive=False):
            if "wishlist-button" in (span.get("class") or []):
                continue
            token = cls._text(span)
            if not token:
                continue
            if raw.sku is None and _RE_SKU.fullmatch(token):
                raw.sku = token
            elif (ml := cls._parse_size_to_ml(token)) is not None:
                raw.size_raw = token
                raw.bottle_size_ml = ml
            elif (abv := cls._parse_abv(token)) is not None:
                raw.abv = abv

    @classmethod
    def _apply_price_block(cls, raw: RawWine, scope: Tag) -> None:
        base = scope.select_one(".base-price")
        if base is not None and raw.price is None:
            raw.price = cls._parse_price(cls._text(base))

        ex_vat = scope.select_one(".ex-vat-price")
        if ex_vat is not None:
            raw.price_ex_vat = cls._parse_price(cls._text(ex_vat))

        was = scope.select_one(
            "del, s, [class*='was'], [class*='original'], [class*='strike'], [class*='rrp']"
        )
        if was is not None:
            was_price = cls._parse_price(cls._text(was))
            if was_price is not None and (raw.price is None or was_price > raw.price):
                raw.on_sale = True
                raw.sale_was_price = was_price

    @classmethod
    def _apply_attributes(cls, raw: RawWine, scope: Tag) -> None:
        """Read the ``.attribute-item`` rows (vintage/colour/producer/region/grape)."""
        for item in scope.select(".attribute-item"):
            classes = item.get("class") or []
            kind = next((c for c in classes if c != "attribute-item"), None)
            link_texts = [t for a in item.select("a") if (t := cls._text(a))]
            text = cls._text(item)

            match kind:
                case "vintage":
                    raw.vintage = cls._parse_vintage(link_texts[0] if link_texts else text)
                case "colour" | "color":
                    raw.color = _COLOR_BY_LABEL.get(text.strip().casefold())
                case "producer":
                    raw.producer = link_texts[0] if link_texts else (text or None)
                case "region":
                    raw.region_raw = text or None
                    if link_texts:
                        raw.region = link_texts[0]
                        if len(link_texts) >= 2:
                            raw.country = link_texts[-1]
                case "sub-group":
                    raw.sub_region = link_texts[0] if link_texts else (text or None)
                case "grape":
                    raw.grapes = link_texts or ([text] if text else [])
                case _:
                    continue

    @classmethod
    def _parse_scores(cls, scope: Tag) -> list[CriticScore]:
        return [
            score
            for header in scope.select("h4.panel-title")
            if (score := cls._parse_score_header(cls._text(header))) is not None
        ]

    @classmethod
    def _detect_in_bond(cls, scope: Tag) -> bool:
        if scope.select_one("[class*='bond']") is not None:
            return True
        heading = cls._text(scope.select_one(".product-heading"))
        return "in bond" in heading.casefold()

    # Dietary / production badges, keyed by the ``product__badge-<suffix>`` CSS
    # class on the product's own badge block. We deliberately match the
    # ``product__badge-*`` class, NOT the ``bd_*`` teaser class: the latter is
    # reused on "related products" carousel tiles and would flag a wine because a
    # *neighbour* is vegan.
    _DIET_BADGES = (
        ("vegan", "is_vegan"),
        ("organic", "is_organic"),
        ("kosher", "is_kosher"),
        ("alcohol_free", "is_alcohol_free"),
    )

    def _apply_diet_badges(self, raw: RawWine) -> None:
        """Set dietary flags from the main product's own ``product__badge-*`` block."""
        scope = self._soup.select_one(".product--full") or self._soup
        for suffix, field in self._DIET_BADGES:
            if scope.select_one(f".product__badge-{suffix}") is not None:
                setattr(raw, field, True)

    def _breadcrumb_section(self) -> str | None:
        """The first non-Home breadcrumb link, e.g. 'Wines' or 'Spirits'."""
        crumb = self._soup.select_one("ol.breadcrumb")
        if crumb is None:
            return None
        return next(
            (
                text
                for link in crumb.select("li a")
                if (text := self._text(link)) and text.casefold() != "home"
            ),
            None,
        )

    # ----------------------------------------------------------------------- #
    # Field-level helpers (pure; no page state)                               #
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _parse_price(value: Any) -> float | None:
        if value is None:
            return None
        match = _RE_PRICE.search(str(value))
        if match is None:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def _parse_size_to_ml(token: str) -> int | None:
        match = _RE_SIZE.fullmatch(token)
        if match is None:
            return None
        value = float(match.group(1))
        match match.group(2).lower():
            case "cl":
                return round(value * 10)
            case "ml":
                return round(value)
            case _:  # litres
                return round(value * 1000)

    @staticmethod
    def _parse_abv(token: str) -> float | None:
        match = _RE_ABV.fullmatch(token)
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_vintage(value: str | None) -> int | None:
        if not value:
            return None
        match = _RE_VINTAGE.search(value)
        return int(match.group(0)) if match else None

    @staticmethod
    def _parse_int(text: str | None) -> int | None:
        if not text:
            return None
        match = _RE_INT.search(text)
        return int(match.group(0)) if match else None

    @staticmethod
    def _parse_score_header(text: str | None) -> CriticScore | None:
        if not text:
            return None
        match = _RE_SCORE.match(text)
        if match is None:
            return None
        try:
            score = float(match.group(1))
        except ValueError:
            return None
        critic = match.group(2).strip()
        scale = 20 if critic.casefold() in _SCALE_20_CRITICS else 100
        return CriticScore(critic=critic, score=score, scale=scale)

    @staticmethod
    def _map_availability(value: Any) -> Availability | None:
        if not value:
            return None
        key = str(value).rsplit("/", 1)[-1].strip().casefold()
        return _AVAILABILITY_BY_SCHEMA.get(key)

    @staticmethod
    def _slug_from_url(url: str) -> str:
        path = urlparse(url).path.rstrip("/")
        return path.rsplit("/", 1)[-1] or path

    @staticmethod
    def _text(node: Tag | None) -> str:
        if node is None:
            return ""
        return _RE_WS.sub(" ", node.get_text(" ")).strip()

    @staticmethod
    def _clean(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _collapse(value: Any) -> str | None:
        if value is None:
            return None
        text = _RE_WS.sub(" ", str(value)).strip()
        return text or None


def parse_product(html: str, url: str) -> RawWine | None:
    """Parse one product page; return ``None`` if it is not a product page."""
    return ProductParser.parse(html, url)
