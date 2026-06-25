"""Translate :class:`WineFilters` into a Qdrant payload filter (I-5).

The hard constraints extracted by query understanding (I-4) become exact
payload-index filters here, so "red Bordeaux under £50" is genuinely filtered at
the index rather than left to semantic similarity. This lives in
``vector_store/`` (it emits Qdrant model objects) so the ``retrieval/`` package
stays free of Qdrant imports, mirroring :mod:`vector_store.payload`.

Semantics: conditions across distinct fields are ANDed (``Filter.must``); within
a single multi-valued field they are ORed (``MatchAny``). We never use ``should``
— it is a soft/optional match and would not actually exclude anything, defeating
the point of hard filters.
"""

from __future__ import annotations

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

from hedonism_assistant.models.query import PriceRange, VintageRange, WineFilters
from hedonism_assistant.vector_store.payload import MAX_CRITIC_SCORE_FIELD


def _range_condition(key: str, bounds: PriceRange | VintageRange) -> FieldCondition | None:
    """Build a ``gte``/``lte`` range condition from whichever bounds are present.

    Note a vintage range condition also excludes non-vintage wines (their
    ``vintage`` payload is absent) — intended: "around 2015" should not surface NV
    bottles.
    """
    if bounds.min is None and bounds.max is None:
        return None
    return FieldCondition(key=key, range=Range(gte=bounds.min, lte=bounds.max))


def build_qdrant_filter(filters: WineFilters) -> Filter | None:
    """Map :class:`WineFilters` to a Qdrant :class:`Filter`, or ``None`` if empty."""
    must: list[FieldCondition] = []

    # Enum lists: serialise to the string values stored in the payload (the
    # payload was written via ``model_dump(mode="json")``).
    if filters.category:
        must.append(
            FieldCondition(key="category", match=MatchAny(any=[c.value for c in filters.category]))
        )
    if filters.color:
        must.append(
            FieldCondition(key="color", match=MatchAny(any=[c.value for c in filters.color]))
        )

    # Free-text keyword lists (already taxonomy-validated upstream).
    for key in ("country", "region", "sub_region", "grapes"):
        values: list[str] = getattr(filters, key)
        if values:
            must.append(FieldCondition(key=key, match=MatchAny(any=values)))

    if filters.vintage_range is not None:
        condition = _range_condition("vintage", filters.vintage_range)
        if condition is not None:
            must.append(condition)
    if filters.price_range is not None:
        condition = _range_condition("price", filters.price_range)
        if condition is not None:
            must.append(condition)

    # Critic score is indexed on the unified 100-pt field; wines with no scores
    # omit that key entirely, so a ``gte`` bound naturally excludes them.
    if filters.min_critic_score is not None:
        must.append(
            FieldCondition(key=MAX_CRITIC_SCORE_FIELD, range=Range(gte=filters.min_critic_score))
        )

    if filters.bottle_size_ml is not None:
        must.append(
            FieldCondition(key="bottle_size_ml", match=MatchValue(value=filters.bottle_size_ml))
        )
    if filters.in_bond is not None:
        must.append(FieldCondition(key="in_bond", match=MatchValue(value=filters.in_bond)))

    return Filter(must=must) if must else None
