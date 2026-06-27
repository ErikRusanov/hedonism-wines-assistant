"""Canonical wine product card and its retrieval-time wrapper."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class WineCategory(StrEnum):
    """Top-level catalogue category (the site's ``Category`` facet).

    Orthogonal to :class:`WineColor`: a sparkling or sweet wine still has a
    colour. Scope is the ``/wines`` section only (spirits, beer and gift cards
    are out of scope).
    """

    STILL = "still"
    SPARKLING = "sparkling"
    SWEET = "sweet"
    FORTIFIED = "fortified"


class WineColor(StrEnum):
    """Wine colour/type (the site's ``Type`` facet). Absent for some wines."""

    RED = "red"
    WHITE = "white"
    ROSE = "rose"


class Availability(StrEnum):
    """Stock status derived from the product card and JSON-LD ``offers``."""

    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"


class CriticScore(BaseModel):
    """A single critic rating.

    The catalogue mixes scales (Parker/Vinous on 100, Jancis Robinson on 20),
    so ``scale`` is explicit and comparisons must normalise against it.
    """

    critic: str = Field(description="Critic or publication, e.g. 'Vinous', 'Jancis Robinson'.")
    score: float
    scale: int = Field(default=100, description="Maximum of the scale (100 or 20).")
    reviewer: str | None = Field(default=None, description="Named reviewer, e.g. 'Neal Martin'.")
    review_date: str | None = Field(
        default=None,
        description="Raw review date as shown on the page; not normalised to a date type.",
    )


class Wine(BaseModel):
    """The canonical product card.

    Single source of truth produced by the data track (scrape → normalize →
    enrich) and consumed by indexing and serving. Optional fields reflect that
    the catalogue is not uniformly populated; coverage is tracked as a quality
    metric during scraping.

    Field provenance worth pinning down for the scraper: ``id`` (SKU),
    ``price``, ``currency``, ``availability`` and ``image_url`` come from the
    page's JSON-LD; everything else (producer, vintage, region, grapes, colour,
    abv, size, critic scores) is parsed from the rendered HTML spec block.
    Note ``producer`` must come from that HTML, *not* JSON-LD ``brand`` — which
    is the retailer ("Hedonism Wines"), not the winery.
    """

    id: str = Field(description="Stable identifier: the catalogue SKU, e.g. 'HED33786'.")
    slug: str = Field(description="URL slug, e.g. 'pichon-lalande-2020'.")
    name: str
    url: HttpUrl

    category: WineCategory
    color: WineColor | None = None

    producer: str | None = None
    country: str | None = None
    region: str | None = None
    sub_region: str | None = Field(default=None, description="Appellation / sub-group.")
    classification: str | None = Field(default=None, description="e.g. 'Second Growth'.")
    vintage: int | None = Field(default=None, description="Vintage year; None for NV or unknown.")
    grapes: list[str] = Field(default_factory=list, description="Variety or blend components.")

    abv: float | None = Field(default=None, description="Alcohol by volume, percent.")
    bottle_size_ml: int = Field(description="Bottle volume in millilitres (75cl -> 750).")
    format_name: str | None = Field(default=None, description="e.g. 'Half', 'Magnum'.")

    price: float = Field(description="Price including VAT, in ``currency``.")
    price_ex_vat: float | None = None
    currency: str = "GBP"
    on_sale: bool = False
    sale_was_price: float | None = Field(
        default=None, description="Pre-discount price when on sale."
    )
    in_bond: bool = Field(default=False, description="Sold in bond (duty/VAT not yet paid).")

    # Dietary / production flags, parsed from the product's own ``product__badge-*``
    # markers. Default False means "not flagged" (i.e. unknown), never a positive
    # claim of the opposite — absence of a vegan badge does not assert non-vegan.
    is_vegan: bool = Field(default=False, description="Carries the 'Vegan' badge.")
    is_organic: bool = Field(default=False, description="Carries the 'Organic' badge.")
    is_kosher: bool = Field(default=False, description="Carries the 'Kosher' badge.")
    is_alcohol_free: bool = Field(
        default=False, description="Carries the '0% / alcohol-free' badge."
    )

    availability: Availability = Availability.IN_STOCK
    stock_qty: int | None = Field(default=None, description="Units left when a count is shown.")

    critic_scores: list[CriticScore] = Field(default_factory=list)

    tasting_notes: str | None = Field(
        default=None,
        description="Editorial description; copyrighted source text — store sparingly.",
    )
    image_url: HttpUrl | None = None

    # Enrichment tags (I-2). Optional and empty by default so the deterministic
    # pipeline never depends on them; populated only when LLM enrichment is on.
    # They serve the "pairing" intent and style-based retrieval and are folded
    # into both the embedding text and the filterable payload.
    style_tags: list[str] = Field(
        default_factory=list,
        description="Descriptive style/occasion tags, e.g. 'full-bodied', 'oaked', 'mineral'.",
    )
    food_pairings: list[str] = Field(
        default_factory=list,
        description="Dishes the wine suits, e.g. 'roast lamb', 'hard cheese'.",
    )

    embedding_text: str | None = Field(
        default=None,
        description="NL passport text embedded for dense retrieval (set during enrichment).",
    )


class RetrievedWine(BaseModel):
    """A wine returned from retrieval, carrying its relevance scores."""

    wine: Wine
    score: float = Field(description="Fusion score from hybrid retrieval (e.g. RRF).")
    rerank_score: float | None = Field(
        default=None, description="Score assigned by the reranking stage, if applied."
    )
