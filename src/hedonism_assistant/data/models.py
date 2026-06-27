"""The raw scrape record produced by I-1 and consumed by normalization (I-2).

``RawWine`` is intentionally *looser* than the canonical
:class:`~hedonism_assistant.models.wine.Wine`: every field is optional and holds
the value exactly as it appears on the page (plus a few ``*_raw`` provenance
fields). Canonicalisation -- mapping the breadcrumb section and tasting notes to
a :class:`~hedonism_assistant.models.wine.WineCategory`, normalising regions and
grape varieties, deduplicating vintages -- is the job of I-2, which turns these
records into ``Wine`` cards. Keeping the scrape output permissive means a single
malformed field never discards an otherwise-good product.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hedonism_assistant.models.wine import Availability, CriticScore, WineColor


class RawWine(BaseModel):
    """One catalogue product as scraped, before normalization/enrichment."""

    # --- Identity & provenance ---
    url: str
    slug: str
    sku: str | None = Field(
        default=None, description="Catalogue SKU from JSON-LD, e.g. 'HED27711'."
    )
    name: str | None = None
    section: str | None = Field(
        default=None,
        description="Breadcrumb section the product sits under ('Wines', 'Spirits', ...).",
    )

    # --- Taxonomy (raw text; canonicalised in I-2) ---
    color: WineColor | None = None
    producer: str | None = None
    region: str | None = None
    sub_region: str | None = Field(
        default=None, description="Appellation / sub-group as shown, e.g. 'Pauillac', 'Pomerol'."
    )
    country: str | None = None
    region_raw: str | None = Field(
        default=None, description="Full region/country label as shown, e.g. 'Tuscany, Italy'."
    )
    vintage: int | None = Field(default=None, description="Vintage year; None for NV or unknown.")
    grapes: list[str] = Field(default_factory=list)

    # --- Physical ---
    abv: float | None = Field(default=None, description="Alcohol by volume, percent.")
    bottle_size_ml: int | None = Field(default=None, description="Bottle volume in millilitres.")
    size_raw: str | None = Field(default=None, description="Raw size token, e.g. '75cl', '300cl'.")

    # --- Commercial ---
    price: float | None = None
    price_ex_vat: float | None = None
    currency: str = "GBP"
    on_sale: bool = False
    sale_was_price: float | None = None
    in_bond: bool = False
    availability: Availability | None = None
    stock_qty: int | None = Field(default=None, description="Units left when a count is shown.")

    # --- Content ---
    critic_scores: list[CriticScore] = Field(default_factory=list)
    tasting_notes: str | None = Field(
        default=None, description="Editorial body text (copyrighted; stored verbatim from JSON-LD)."
    )
    image_url: str | None = None

    # --- Scrape bookkeeping ---
    fetched_at: str | None = Field(default=None, description="ISO-8601 UTC fetch timestamp.")
    from_cache: bool = Field(default=False, description="True if served from the on-disk cache.")

    @property
    def is_wine(self) -> bool:
        """Whether the breadcrumb places this product in the wine catalogue."""
        return (self.section or "").strip().casefold() == "wines"
