"""Deterministic guardrail responses — out-of-scope, other drinks, empty retrieval (I-6).

These edges must never reach the generation model: an off-domain question has no
wines to ground on, a non-wine drink belongs to a different catalogue, and an empty
result set has nothing to answer from. All are handled here with fixed copy and
rule-derived follow-ups, which keeps them fast, predictable, and free of an LLM call
(handy while the API is unfunded).

The empty-retrieval suggestions read the *active* filters and propose relaxing the
most constraining ones first, so the user gets concrete next steps ("raise the
budget", "broaden beyond Bordeaux") rather than a generic apology.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

from hedonism_assistant.models.query import WineFilters

OUT_OF_SCOPE_MESSAGE: Final = (
    "I'm the Hedonism Wines assistant, so I can only help with the wine catalogue — "
    "finding bottles, styles, regions, vintages, pairings and gift ideas. For orders, "
    "delivery, stock or account questions, please visit hedonism.co.uk or ask their team."
)

EMPTY_RETRIEVAL_MESSAGE: Final = (
    "I couldn't find any wines in the catalogue matching those criteria. "
    "You could relax some of the constraints and try again."
)


def other_drinks_message(spirits_url: str) -> str:
    """Redirect a non-wine drink request to Hedonism's spirits range.

    The assistant is grounded in the wine catalogue only, so it points the user to
    the right place rather than guessing at bottles it cannot see.
    """
    return (
        "I only know the Hedonism wine catalogue, so I can't advise on spirits, beer "
        "or other drinks. Hedonism does stock a huge range of them, though — you can "
        f"browse it at {spirits_url}. Happy to help with any wine instead."
    )


# Offered when the user is off-domain, to steer them back toward what we can answer.
_OUT_OF_SCOPE_SUGGESTIONS: Final = (
    "Ask for a wine by style, region or grape, e.g. 'a bold red Bordeaux under £80'.",
    "Ask what to drink with a dish, e.g. 'a white to pair with roast chicken'.",
    "Ask about a specific bottle's vintage, price or critic scores.",
)

_GENERIC_RELAXATION: Final = "Try describing the wine differently or with broader terms."


def out_of_scope_suggestions(*, limit: int) -> list[str]:
    """Static nudges back toward in-scope wine questions, capped at ``limit``."""
    return list(_OUT_OF_SCOPE_SUGGESTIONS[: max(limit, 0)])


def empty_retrieval_suggestions(filters: WineFilters, *, limit: int) -> list[str]:
    """Propose relaxations of the active filters, most-constraining first.

    Falls back to a generic broadening hint when the query carried no hard
    filters (a purely semantic miss).
    """
    suggestions = list(_relaxations(filters)) or [_GENERIC_RELAXATION]
    return suggestions[: max(limit, 0)]


def _relaxations(filters: WineFilters) -> Iterator[str]:
    """Yield human relaxation hints for whichever filters are set."""
    if filters.price_range and filters.price_range.max is not None:
        yield f"Raise your budget above £{filters.price_range.max:.0f}."
    if filters.price_range and filters.price_range.min is not None:
        yield f"Lower the minimum price below £{filters.price_range.min:.0f}."
    if filters.min_critic_score is not None:
        yield f"Lower the critic-score threshold below {filters.min_critic_score:.0f}/100."
    if filters.sub_region:
        yield f"Look beyond the {_join(filters.sub_region)} appellation."
    if filters.region:
        yield f"Broaden beyond {_join(filters.region)}."
    if filters.country:
        yield f"Include countries other than {_join(filters.country)}."
    if filters.grapes:
        yield f"Allow grapes other than {_join(filters.grapes)}."
    if filters.vintage_range:
        yield "Allow other vintages."
    if filters.color:
        yield "Consider other wine colours."
    if filters.category:
        yield "Consider other wine styles (sparkling, sweet, fortified)."
    if filters.bottle_size_ml is not None:
        yield "Allow other bottle sizes."
    if filters.in_bond is not None:
        yield "Include both in-bond and duty-paid bottles."


def _join(values: list[str]) -> str:
    """Render a small string list for prose, e.g. ``'Bordeaux and Burgundy'``."""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]
