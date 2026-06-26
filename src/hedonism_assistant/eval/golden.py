"""Golden set and predicate-based relevance for the eval harness (I-8).

A golden case pairs a natural-language question with a definition of which wines
*should* surface. Relevance is defined two ways, in priority order:

1. **Explicit ids** — ``expected_wine_ids`` pin specific SKUs (used for factual
   "tell me about this bottle" questions where one card is the answer).
2. **Predicate** — a :class:`WineFilters` ``relevance`` spec: a retrieved wine is
   relevant iff it satisfies *every* set field (red AND Bordeaux AND ≤ £50 …).

The predicate form is the default because it is authored against stable
catalogue attributes (region/colour/price/grape) rather than brittle SKUs the
re-capture can change, and it reuses the same vocabulary the query parser emits.
``matches`` mirrors the conditions in :func:`vector_store.filters.build_qdrant_filter`.

Guardrail cases carry ``expected_fallback`` instead and are scored separately
(did the pipeline take the expected branch), not via retrieval metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.query import PriceRange, QueryIntent, VintageRange, WineFilters
from hedonism_assistant.models.wine import Wine
from hedonism_assistant.vector_store.payload import normalize_critic_score

logger = get_logger(__name__)

# The guardrail branch a case expects the pipeline to take, when it is not a
# normal retrieval case. Mirrors the short-circuits in ``ChatService``.
type ExpectedFallback = Literal["other_drinks", "out_of_scope", "empty"]


class GoldenCase(BaseModel):
    """One evaluation question plus its ground-truth relevance definition."""

    id: str
    question: str
    expected_intent: QueryIntent | None = None
    # Predicate relevance: a wine is relevant iff it satisfies every set field.
    relevance: WineFilters = Field(default_factory=WineFilters)
    # Explicit id pin: overrides the predicate when non-empty.
    expected_wine_ids: list[str] = Field(default_factory=list)
    # Optional reference answer (not required by the lean judge metrics).
    reference_answer: str | None = None
    # Set for guardrail cases instead of a retrieval relevance spec.
    expected_fallback: ExpectedFallback | None = None


def load_golden(path: str | Path) -> list[GoldenCase]:
    """Load golden cases from JSONL, skipping blank lines and logging bad rows."""
    cases: list[GoldenCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                cases.append(GoldenCase.model_validate_json(stripped))
            except ValidationError as exc:
                logger.warning("golden_parse_failed", error=str(exc))
    return cases


def relevant_ids(case: GoldenCase, retrieved: list[Wine]) -> set[str]:
    """The relevant ids for a case among the ``retrieved`` wines.

    Explicit ``expected_wine_ids`` win when present; otherwise a wine counts as
    relevant iff it satisfies the case's predicate. An empty predicate with no
    id pin yields no relevant ids (such a case should carry ``expected_fallback``).
    """
    if case.expected_wine_ids:
        pinned = set(case.expected_wine_ids)
        return {w.id for w in retrieved if w.id in pinned}
    return {w.id for w in retrieved if matches(w, case.relevance)}


def matches(wine: Wine, spec: WineFilters) -> bool:
    """Whether ``wine`` satisfies every set field of ``spec``.

    Mirrors :func:`vector_store.filters.build_qdrant_filter`: distinct fields are
    ANDed; a multi-valued field is an OR-membership test. Unset fields impose no
    constraint, so an empty spec matches everything.
    """
    return (
        _in_enum(wine.category, spec.category)
        and _in_enum(wine.color, spec.color)
        and _in_str(wine.producer, spec.producer)
        and _in_str(wine.country, spec.country)
        and _in_str(wine.region, spec.region)
        and _in_str(wine.sub_region, spec.sub_region)
        and _intersects(wine.grapes, spec.grapes)
        and _in_range(wine.vintage, spec.vintage_range)
        and _in_range(wine.price, spec.price_range)
        and _eq(wine.bottle_size_ml, spec.bottle_size_ml)
        and _eq(wine.in_bond, spec.in_bond)
        and _meets_score(wine, spec.min_critic_score)
    )


def _in_enum(value: object | None, allowed: list[object]) -> bool:
    return not allowed or value in allowed


def _in_str(value: str | None, allowed: list[str]) -> bool:
    if not allowed:
        return True
    if value is None:
        return False
    folded = {a.casefold() for a in allowed}
    return value.casefold() in folded


def _intersects(values: list[str], allowed: list[str]) -> bool:
    if not allowed:
        return True
    folded = {a.casefold() for a in allowed}
    return any(v.casefold() in folded for v in values)


def _in_range(value: float | None, bounds: PriceRange | VintageRange | None) -> bool:
    if bounds is None or (bounds.min is None and bounds.max is None):
        return True
    # A range excludes wines lacking the value (e.g. NV under a vintage range),
    # matching the index-side filter semantics.
    if value is None:
        return False
    if bounds.min is not None and value < bounds.min:
        return False
    return not (bounds.max is not None and value > bounds.max)


def _eq(value: object, expected: object | None) -> bool:
    return expected is None or value == expected


def _meets_score(wine: Wine, minimum: float | None) -> bool:
    if minimum is None:
        return True
    best = max(
        (normalize_critic_score(s.score, s.scale) for s in wine.critic_scores),
        default=None,
    )
    return best is not None and best >= minimum
