"""Catalogue taxonomy used to validate query-understanding filters.

The query parser is an LLM and will happily invent regions or misspell grapes.
A filter value that does not exist in the catalogue payload would silently zero
out retrieval, so free-text filter dimensions (country, region, sub-region,
grape) are validated against the set of values actually present in the data.

The taxonomy is derived from the indexed :class:`Wine` cards, so it always
reflects the live catalogue. Before any data is indexed (e.g. the serving track
running on fixtures), an empty taxonomy is used and validation degrades to a
pass-through: cleaned values are kept verbatim rather than dropped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from hedonism_assistant.models.wine import Wine


class TaxonomyDimension(StrEnum):
    """Free-text filter dimensions that are validated against the catalogue."""

    COUNTRY = "country"
    REGION = "region"
    SUB_REGION = "sub_region"
    GRAPE = "grape"


@dataclass(frozen=True)
class Taxonomy:
    """Known, canonical values for each free-text filter dimension.

    Matching is case-insensitive: an LLM may emit ``"bordeaux"`` while the
    catalogue stores ``"Bordeaux"``. Each dimension keeps a lower-cased lookup
    that maps back to the canonical spelling stored in the payload.
    """

    countries: frozenset[str] = frozenset()
    regions: frozenset[str] = frozenset()
    sub_regions: frozenset[str] = frozenset()
    grapes: frozenset[str] = frozenset()

    _lookup: dict[TaxonomyDimension, dict[str, str]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        members = {
            TaxonomyDimension.COUNTRY: self.countries,
            TaxonomyDimension.REGION: self.regions,
            TaxonomyDimension.SUB_REGION: self.sub_regions,
            TaxonomyDimension.GRAPE: self.grapes,
        }
        for dimension, values in members.items():
            self._lookup[dimension] = {v.casefold(): v for v in values}

    @classmethod
    def from_wines(cls, wines: Iterable[Wine]) -> Taxonomy:
        """Build a taxonomy from indexed wine cards."""
        countries: set[str] = set()
        regions: set[str] = set()
        sub_regions: set[str] = set()
        grapes: set[str] = set()
        for wine in wines:
            if wine.country:
                countries.add(wine.country)
            if wine.region:
                regions.add(wine.region)
            if wine.sub_region:
                sub_regions.add(wine.sub_region)
            grapes.update(g for g in wine.grapes if g)
        return cls(
            countries=frozenset(countries),
            regions=frozenset(regions),
            sub_regions=frozenset(sub_regions),
            grapes=frozenset(grapes),
        )

    def canonicalize(self, dimension: TaxonomyDimension, values: Iterable[str]) -> list[str]:
        """Map ``values`` onto canonical catalogue spellings, dropping unknowns.

        Duplicates and blanks are removed and order is preserved. When the
        dimension is empty (no data indexed yet) validation cannot be performed,
        so cleaned values are passed through unchanged.
        """
        lookup = self._lookup[dimension]
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            if not isinstance(raw, str):
                continue
            cleaned = raw.strip()
            if not cleaned:
                continue
            canonical = cleaned if not lookup else lookup.get(cleaned.casefold())
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            out.append(canonical)
        return out
