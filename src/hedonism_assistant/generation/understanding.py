"""Render a parsed query as human-readable "understanding" chips.

A pure, deterministic projection of :class:`ParsedQuery` into short labels the UI
shows as read-only pills (e.g. ``Red``, ``Bordeaux``, ``under £50``). It surfaces
the hard filters the assistant extracted so the user sees they were understood; it
never calls a model.
"""

from __future__ import annotations

from hedonism_assistant.models.query import ParsedQuery, PriceRange, VintageRange

# Catalogue currency symbols for price chips; falls back to the code itself.
_CURRENCY_SYMBOL = {"GBP": "£", "EUR": "€", "USD": "$"}


def _price_chip(price: PriceRange, symbol: str) -> str | None:
    lo, hi = price.min, price.max
    if lo is not None and hi is not None:
        return f"{symbol}{lo:g}–{symbol}{hi:g}"
    if hi is not None:
        return f"under {symbol}{hi:g}"
    if lo is not None:
        return f"over {symbol}{lo:g}"
    return None


def _vintage_chip(vintage: VintageRange) -> str | None:
    lo, hi = vintage.min, vintage.max
    if lo is not None and hi is not None:
        return str(lo) if lo == hi else f"{lo}–{hi}"
    if hi is not None:
        return f"up to {hi}"
    if lo is not None:
        return f"from {lo}"
    return None


def filters_to_chips(parsed: ParsedQuery, currency: str = "GBP") -> list[str]:
    """Project the hard filters of ``parsed`` into short, ordered chip labels."""
    filters = parsed.filters
    symbol = _CURRENCY_SYMBOL.get(currency, currency + " ")
    chips: list[str] = []

    # Colour and category first (most salient), then place, producer, grape.
    chips.extend(c.capitalize() for c in filters.color)
    chips.extend(c.capitalize() for c in filters.category)
    chips.extend(filters.sub_region)
    chips.extend(filters.region)
    chips.extend(filters.country)
    chips.extend(filters.producer)
    chips.extend(filters.grapes)

    if filters.price_range and (chip := _price_chip(filters.price_range, symbol)):
        chips.append(chip)
    if filters.vintage_range and (chip := _vintage_chip(filters.vintage_range)):
        chips.append(chip)
    if filters.min_critic_score is not None:
        chips.append(f"{filters.min_critic_score:g}+ pts")
    if filters.bottle_size_ml:
        chips.append(f"{filters.bottle_size_ml / 1000:g}L")
    if filters.in_bond:
        chips.append("in bond")

    # De-duplicate case-insensitively while preserving order (a producer can echo
    # a region, etc.).
    seen: set[str] = set()
    unique: list[str] = []
    for chip in chips:
        key = chip.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(chip)
    return unique
