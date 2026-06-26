"""Query understanding (self-query): natural language -> :class:`ParsedQuery`.

A cheap utility model splits a user message into a ``semantic_query`` (for
dense/sparse search), hard ``WineFilters`` (which become Qdrant payload-index
filters) and an ``intent``. Pulling "red Bordeaux under £50" out as
``color=red``, ``region=Bordeaux``, ``price_range.max=50`` is what lets the
catalogue be filtered exactly instead of searched as free text.

The stage is built to never fail the request: any chain/parse error or
off-domain input degrades to a soft fallback on pure semantics
(``confident=False``), and filter coercion drops individual bad values rather
than rejecting the whole parse.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import lru_cache

from openai.types.chat import ChatCompletionMessageParam

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.llm.json_output import loads_json
from hedonism_assistant.llm.openrouter import OpenRouterClient, get_openrouter_client
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.query import (
    ParsedQuery,
    PriceRange,
    QueryIntent,
    VintageRange,
    WineFilters,
)
from hedonism_assistant.models.wine import WineCategory, WineColor
from hedonism_assistant.retrieval.taxonomy import Taxonomy, TaxonomyDimension

logger = get_logger(__name__)

# Casefolded value -> member, so LLM casing ("RED", "Still") maps without a
# try/except per value. StrEnum members are their own string values.
_COLOR_BY_VALUE: dict[str, WineColor] = {c.casefold(): c for c in WineColor}
_CATEGORY_BY_VALUE: dict[str, WineCategory] = {c.casefold(): c for c in WineCategory}

# Raw JSON key -> the taxonomy dimension that validates it.
_TAXONOMY_DIMENSIONS: dict[str, TaxonomyDimension] = {
    "country": TaxonomyDimension.COUNTRY,
    "region": TaxonomyDimension.REGION,
    "sub_region": TaxonomyDimension.SUB_REGION,
    "grapes": TaxonomyDimension.GRAPE,
}

_SYSTEM_PROMPT = """\
You are a query-understanding component for a wine-catalogue search engine.
Turn the user's message into a single JSON object that splits it into a semantic
search query, hard metadata filters, and an intent.

Return ONLY a JSON object with this shape (omit fields you cannot fill):
{
  "semantic_query": string,        // descriptive text for semantic search
  "intent": string,                // one of: recommendation, factual, pairing,
                                   //         comparison, out_of_scope
  "filters": {
    "category": [string],          // still, sparkling, sweet, fortified
    "color": [string],             // red, white, rose
    "country": [string],           // e.g. France, Italy
    "region": [string],            // e.g. Bordeaux, Tuscany
    "sub_region": [string],        // appellation, e.g. Pauillac
    "grapes": [string],            // e.g. Pinot Noir, Nebbiolo
    "vintage_range": {"min": int, "max": int},
    "price_range": {"min": number, "max": number},
    "bottle_size_ml": int,         // e.g. 1500 for a magnum
    "min_critic_score": number,    // normalised to a 100-pt scale
    "in_bond": bool
  }
}

Rules:
- Extract every HARD constraint into filters: price ("under £50" ->
  price_range.max=50), vintage ("2015", "before 2010"), region/country/
  sub-region, grape variety, colour, category, bottle size, in-bond, critic
  score ("90+ points" -> min_critic_score=90).
- Leave DESCRIPTIVE wishes (style, occasion, food to pair with, mood) in
  semantic_query. Do not turn them into filters.
- Classify intent. Use "pairing" when the user asks what to drink with food,
  "comparison" when contrasting options, "factual" for specific fact lookups,
  "recommendation" otherwise.
- For non-wine requests, set intent to "out_of_scope", leave filters empty and
  put a short restatement of the message in semantic_query.

Examples:
User: "red Bordeaux under £50"
{"semantic_query": "red Bordeaux", "intent": "recommendation",
 "filters": {"color": ["red"], "region": ["Bordeaux"], "price_range": {"max": 50}}}

User: "elegant Burgundy pinot noir to go with duck, around 2015"
{"semantic_query": "elegant Burgundy to go with duck", "intent": "pairing",
 "filters": {"region": ["Burgundy"], "grapes": ["Pinot Noir"],
             "vintage_range": {"min": 2015, "max": 2015}}}

User: "what's the weather like today?"
{"semantic_query": "what's the weather like today", "intent": "out_of_scope",
 "filters": {}}
"""


class QueryParser:
    """Parse a user message into a :class:`ParsedQuery` via the utility model."""

    def __init__(
        self,
        client: OpenRouterClient,
        settings: Settings,
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        # An empty taxonomy degrades free-text validation to pass-through, so
        # the parser works before any data is indexed.
        self._taxonomy = taxonomy or Taxonomy()

    def set_taxonomy(self, taxonomy: Taxonomy) -> None:
        """Replace the validation taxonomy (late-injected from the live index in I-7).

        The cached parser singleton is built with an empty pass-through taxonomy;
        the serving layer loads the real catalogue taxonomy on startup and calls
        this so subsequent filter coercion validates against indexed values.
        """
        self._taxonomy = taxonomy

    async def parse(self, message: str) -> ParsedQuery:
        """Parse ``message``; never raises, always returns a usable query."""
        if not self._settings.query_parsing_enabled:
            return self._pure_semantic(message, confident=True)

        try:
            raw = loads_json(await self._complete(message))
        except Exception as exc:  # noqa: BLE001 - resilience boundary: parsing must never fail a request
            logger.warning("query_parse_failed", error=str(exc))
            return self._pure_semantic(message, confident=False)

        if not isinstance(raw, dict):
            logger.warning("query_parse_non_object", payload_type=type(raw).__name__)
            return self._pure_semantic(message, confident=False)

        return self._coerce(raw, message)

    async def _complete(self, message: str) -> str:
        """Call the utility model and return its raw (expected-JSON) content."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        return await self._client.chat(
            messages,
            model=self._settings.utility_model,
            fallback_models=self._settings.utility_fallback_models,
            temperature=self._settings.query_parse_temperature,
            response_format={"type": "json_object"},
        )

    def _coerce(self, raw: dict[str, object], message: str) -> ParsedQuery:
        """Build a :class:`ParsedQuery` from raw JSON; never raises."""
        semantic = raw.get("semantic_query")
        if not isinstance(semantic, str) or not semantic.strip():
            semantic = message
        return ParsedQuery(
            semantic_query=semantic,
            filters=self._coerce_filters(raw.get("filters")),
            intent=self._as_enum(raw.get("intent"), QueryIntent, QueryIntent.RECOMMENDATION),
            confident=True,
        )

    def _coerce_filters(self, raw: object) -> WineFilters:
        """Coerce a raw filters mapping; one bad value never sinks the parse."""
        if not isinstance(raw, dict):
            return WineFilters()

        canonical = {
            key: self._taxonomy.canonicalize(dimension, self._as_str_list(raw.get(key)))
            for key, dimension in _TAXONOMY_DIMENSIONS.items()
        }
        in_bond = raw.get("in_bond")
        return WineFilters(
            category=self._coerce_enum_list(raw.get("category"), _CATEGORY_BY_VALUE),
            color=self._coerce_enum_list(raw.get("color"), _COLOR_BY_VALUE),
            **canonical,
            vintage_range=self._build_range(raw.get("vintage_range"), VintageRange, int),
            price_range=self._build_range(raw.get("price_range"), PriceRange, float),
            bottle_size_ml=self._coerce_number(raw.get("bottle_size_ml"), int),
            min_critic_score=self._coerce_number(raw.get("min_critic_score"), float),
            in_bond=in_bond if isinstance(in_bond, bool) else None,
        )

    @staticmethod
    def _pure_semantic(message: str, *, confident: bool) -> ParsedQuery:
        """Fallback parse: the whole message as the semantic query, no filters."""
        return ParsedQuery(
            semantic_query=message,
            intent=QueryIntent.RECOMMENDATION,
            confident=confident,
        )

    @staticmethod
    def _as_str_list(value: object) -> list[str]:
        """Normalise a raw field into a list of strings (tolerating a bare string)."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []

    @staticmethod
    def _as_enum[E: StrEnum](value: object, enum: type[E], default: E) -> E:
        """Coerce ``value`` to an ``enum`` member, falling back to ``default``."""
        try:
            return enum(value)
        except ValueError:
            return default

    @staticmethod
    def _coerce_enum_list[E: StrEnum](value: object, by_value: dict[str, E]) -> list[E]:
        """Keep the values that map (case-insensitively) to an enum member."""
        out: list[E] = []
        seen: set[E] = set()
        for raw in QueryParser._as_str_list(value):
            member = by_value.get(raw.strip().casefold())
            if member is not None and member not in seen:
                seen.add(member)
                out.append(member)
        return out

    @staticmethod
    def _coerce_number[N: (int, float)](value: object, cast: Callable[[float], N]) -> N | None:
        """Cast a JSON number to ``int``/``float``; reject bools and non-numerics."""
        match value:
            case bool():
                return None
            case int() | float():
                return cast(value)
            case _:
                return None

    @staticmethod
    def _build_range[R: (VintageRange, PriceRange)](
        raw: object, model: type[R], cast: Callable[[float], int | float]
    ) -> R | None:
        """Build a bounds model when at least one numeric bound is present."""
        if not isinstance(raw, dict):
            return None
        low = QueryParser._coerce_number(raw.get("min"), cast)
        high = QueryParser._coerce_number(raw.get("max"), cast)
        if low is None and high is None:
            return None
        return model(min=low, max=high)


@lru_cache
def get_query_parser() -> QueryParser:
    """Return a cached parser built from the shared client and settings.

    Taxonomy injection (from the live index or fixtures) happens in later
    iterations; here the parser starts with an empty pass-through taxonomy.
    """
    return QueryParser(get_openrouter_client(), get_settings())
