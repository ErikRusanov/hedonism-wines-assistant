"""Optional LLM enrichment for normalized wine cards (I-2).

The deterministic pass in :mod:`hedonism_assistant.data.normalize` produces a
complete, indexable :class:`Wine` on its own. This module adds the *optional*
tags the catalogue does not state outright and that help the "pairing" and
style-based queries the service is built for: a colour when the page omitted it,
a handful of ``style_tags`` (body, sweetness, oak, texture) and ``food_pairings``.

It is gated behind ``settings.enrich_use_llm`` (off by default) so the pipeline
runs fully offline without it. Like the query parser, it is built to never fail
the run: any chain or parse error leaves the card exactly as it came in. Tags are
merged additively -- enrichment never overwrites a value the data already had.
"""

from __future__ import annotations

import asyncio
import json
from typing import Final

from openai.types.chat import ChatCompletionMessageParam

from hedonism_assistant.config import Settings
from hedonism_assistant.llm.openrouter import OpenRouterClient
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.wine import Wine, WineColor

logger = get_logger(__name__)

# Casefolded colour word -> enum member, for resolving the model's free-text colour.
_COLOR_BY_VALUE: Final[dict[str, WineColor]] = {c.value.casefold(): c for c in WineColor}

_SYSTEM_PROMPT = """\
You are a sommelier tagging wines for a catalogue search engine. Given a wine's
facts, return ONLY a JSON object describing it for retrieval:
{
  "color": string,          // red, white or rose -- ONLY if clearly implied; else omit
  "style_tags": [string],   // 3-6 short descriptors: body, sweetness, oak, texture,
                            //   e.g. "full-bodied", "dry", "oaked", "mineral", "tannic"
  "food_pairings": [string] // 3-6 dishes it suits, e.g. "roast lamb", "hard cheese"
}
Use only the facts given and well-established wine knowledge. Do not invent a
producer, region or vintage. Keep every tag lowercase and under four words. If you
are unsure about colour, omit it.
"""


class LlmEnricher:
    """Fill a missing colour and add style/food-pairing tags via the utility model."""

    def __init__(self, client: OpenRouterClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def enrich(self, wine: Wine) -> Wine:
        """Return ``wine`` augmented with LLM tags; never raises."""
        try:
            raw = json.loads(await self._complete(wine))
        except Exception as exc:  # noqa: BLE001 - resilience boundary: enrichment must not fail the run
            logger.warning("enrich_failed", wine_id=wine.id, error=str(exc))
            return wine
        if not isinstance(raw, dict):
            return wine
        return self._merge(wine, raw)

    async def enrich_many(self, wines: list[Wine]) -> list[Wine]:
        """Enrich many wines with bounded concurrency, preserving order."""
        semaphore = asyncio.Semaphore(max(1, self._settings.enrich_llm_concurrency))

        async def _one(wine: Wine) -> Wine:
            async with semaphore:
                return await self.enrich(wine)

        return await asyncio.gather(*(_one(wine) for wine in wines))

    async def _complete(self, wine: Wine) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._wine_facts(wine)},
        ]
        return await self._client.chat(
            messages,
            model=self._settings.utility_model,
            fallback_models=self._settings.utility_fallback_models,
            temperature=self._settings.enrich_llm_temperature,
            response_format={"type": "json_object"},
        )

    @staticmethod
    def _wine_facts(wine: Wine) -> str:
        """Compact fact sheet handed to the model (bounded note excerpt)."""
        facts = {
            "name": wine.name,
            "category": wine.category.value,
            "color": wine.color.value if wine.color else None,
            "region": wine.region,
            "country": wine.country,
            "grapes": wine.grapes,
            "tasting_notes": (wine.tasting_notes or "")[:500] or None,
        }
        return json.dumps({k: v for k, v in facts.items() if v}, ensure_ascii=False)

    def _merge(self, wine: Wine, raw: dict[str, object]) -> Wine:
        """Apply LLM output additively; existing values win."""
        updates: dict[str, object] = {}
        limit = self._settings.enrich_max_tags

        if wine.color is None and (color := self._parse_color(raw.get("color"))) is not None:
            updates["color"] = color
        if style := self._clean_tags(raw.get("style_tags")):
            updates["style_tags"] = self._dedupe(wine.style_tags + style, limit)
        if pairings := self._clean_tags(raw.get("food_pairings")):
            updates["food_pairings"] = self._dedupe(wine.food_pairings + pairings, limit)

        return wine.model_copy(update=updates) if updates else wine

    @classmethod
    def _parse_color(cls, value: object) -> WineColor | None:
        """Resolve the model's free-text colour to an enum member, or ``None``."""
        return _COLOR_BY_VALUE.get(str(value or "").strip().casefold())

    @staticmethod
    def _clean_tags(value: object) -> list[str]:
        """Lowercased, whitespace-stripped, non-empty string tags from raw output."""
        if not isinstance(value, list):
            return []
        return [tag for item in value if isinstance(item, str) and (tag := item.strip().lower())]

    @staticmethod
    def _dedupe(tags: list[str], limit: int) -> list[str]:
        """Order-preserving dedupe, capped at ``limit`` tags."""
        return list(dict.fromkeys(tags))[:limit]
