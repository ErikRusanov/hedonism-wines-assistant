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

from collections.abc import Iterator

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

from hedonism_assistant.models.query import PriceRange, VintageRange, WineFilters
from hedonism_assistant.vector_store.payload import MAX_CRITIC_SCORE_FIELD


def _match_any(key: str, values: list[str]) -> FieldCondition | None:
    """OR-match a keyword field against ``values`` (``None`` when there are none)."""
    return FieldCondition(key=key, match=MatchAny(any=values)) if values else None


def _match_value(key: str, value: int | bool | None) -> FieldCondition | None:
    """Exact-match a scalar field (``None`` when the value is absent).

    ``in_bond=False`` is a real constraint, so the guard tests ``is not None``
    rather than truthiness.
    """
    return None if value is None else FieldCondition(key=key, match=MatchValue(value=value))


def _range(key: str, bounds: PriceRange | VintageRange | None) -> FieldCondition | None:
    """Build a ``gte``/``lte`` range from whichever bounds are present.

    A vintage range also excludes non-vintage wines (their ``vintage`` payload is
    absent) — intended: "around 2015" should not surface NV bottles.
    """
    if bounds is None or (bounds.min is None and bounds.max is None):
        return None
    return FieldCondition(key=key, range=Range(gte=bounds.min, lte=bounds.max))


def _conditions(filters: WineFilters) -> Iterator[FieldCondition | None]:
    """Yield one (possibly ``None``) condition per filterable dimension, in order.

    Enum lists serialise to their string ``value`` — the form stored in the
    payload (written via ``model_dump(mode="json")``).
    """
    yield _match_any("category", [c.value for c in filters.category])
    yield _match_any("color", [c.value for c in filters.color])
    yield _match_any("producer", filters.producer)
    yield _match_any("country", filters.country)
    yield _match_any("region", filters.region)
    yield _match_any("sub_region", filters.sub_region)
    yield _match_any("grapes", filters.grapes)
    yield _range("vintage", filters.vintage_range)
    yield _range("price", filters.price_range)
    # Critic score is indexed on the unified 100-pt field; scoreless wines omit
    # the key entirely, so a ``gte`` bound naturally excludes them.
    if filters.min_critic_score is not None:
        yield FieldCondition(key=MAX_CRITIC_SCORE_FIELD, range=Range(gte=filters.min_critic_score))
    yield _match_value("bottle_size_ml", filters.bottle_size_ml)
    yield _match_value("in_bond", filters.in_bond)
    yield _match_value("is_vegan", filters.is_vegan)
    yield _match_value("is_organic", filters.is_organic)
    yield _match_value("is_kosher", filters.is_kosher)
    yield _match_value("is_alcohol_free", filters.is_alcohol_free)


def build_qdrant_filter(filters: WineFilters) -> Filter | None:
    """Map :class:`WineFilters` to a Qdrant :class:`Filter`, or ``None`` if empty."""
    must = [condition for condition in _conditions(filters) if condition is not None]
    return Filter(must=must) if must else None
