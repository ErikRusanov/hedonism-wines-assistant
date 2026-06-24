"""Structured query representation produced by query understanding (self-query)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from hedonism_assistant.models.wine import WineCategory, WineColor


class QueryIntent(StrEnum):
    """Coarse intent that steers retrieval and answer composition."""

    RECOMMENDATION = "recommendation"
    FACTUAL = "factual"
    PAIRING = "pairing"
    COMPARISON = "comparison"
    OUT_OF_SCOPE = "out_of_scope"


class PriceRange(BaseModel):
    """Inclusive price bounds in the catalogue currency."""

    min: float | None = None
    max: float | None = None


class VintageRange(BaseModel):
    """Inclusive vintage-year bounds (non-vintage wines are excluded by a range)."""

    min: int | None = None
    max: int | None = None


class WineFilters(BaseModel):
    """Hard payload filters extracted from the natural-language query.

    These map directly onto Qdrant payload-index filters, so the catalogue
    taxonomy validation happens against these fields rather than free text.
    """

    category: list[WineCategory] = Field(default_factory=list)
    color: list[WineColor] = Field(default_factory=list)
    country: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    sub_region: list[str] = Field(default_factory=list)
    grapes: list[str] = Field(default_factory=list)
    vintage_range: VintageRange | None = None
    price_range: PriceRange | None = None
    bottle_size_ml: int | None = None
    min_critic_score: float | None = Field(
        default=None, description="Lower bound on any critic score, normalised to a 100-pt scale."
    )
    in_bond: bool | None = None


class ParsedQuery(BaseModel):
    """The output of the query-understanding stage.

    Splits a user message into a semantic query (for dense/sparse search), hard
    metadata filters, and an intent. ``confident`` lets downstream stages fall
    back to pure semantics when parsing is unreliable or the query is off-domain.
    """

    semantic_query: str
    filters: WineFilters = Field(default_factory=WineFilters)
    intent: QueryIntent = QueryIntent.RECOMMENDATION
    confident: bool = True
