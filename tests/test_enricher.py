"""Tests for the optional LLM enrichment step (I-2) with a stub client."""

from collections.abc import Iterable

from hedonism_assistant.config import Settings
from hedonism_assistant.data.enricher import LlmEnricher
from hedonism_assistant.models.wine import Wine, WineCategory, WineColor


class _StubClient:
    """Stands in for OpenRouterClient.chat; returns a canned completion."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def chat(self, messages: Iterable[object], **kwargs: object) -> str:
        self.calls += 1
        return self.payload


def _wine(**overrides: object) -> Wine:
    base: dict[str, object] = {
        "id": "HED1",
        "slug": "x",
        "name": "X",
        "url": "https://hedonism.co.uk/product/x",
        "category": WineCategory.STILL,
        "bottle_size_ml": 750,
        "price": 10.0,
    }
    base.update(overrides)
    return Wine(**base)


async def test_fills_missing_colour_and_tags() -> None:
    client = _StubClient(
        '{"color": "red", "style_tags": ["Full-Bodied", "oaked"], "food_pairings": ["Roast Lamb"]}'
    )
    enricher = LlmEnricher(client, Settings())  # type: ignore[arg-type]

    wine = await enricher.enrich(_wine(color=None))

    assert wine.color is WineColor.RED
    assert wine.style_tags == ["full-bodied", "oaked"]
    assert wine.food_pairings == ["roast lamb"]


async def test_existing_colour_is_not_overwritten() -> None:
    client = _StubClient('{"color": "white"}')
    enricher = LlmEnricher(client, Settings())  # type: ignore[arg-type]

    wine = await enricher.enrich(_wine(color=WineColor.RED))

    assert wine.color is WineColor.RED


async def test_invalid_json_leaves_wine_unchanged() -> None:
    client = _StubClient("not json at all")
    enricher = LlmEnricher(client, Settings())  # type: ignore[arg-type]

    original = _wine(color=WineColor.RED)
    assert await enricher.enrich(original) == original


async def test_enrich_many_preserves_order() -> None:
    client = _StubClient('{"style_tags": ["dry"]}')
    enricher = LlmEnricher(client, Settings())  # type: ignore[arg-type]

    wines = [_wine(id=f"HED{i}", slug=str(i)) for i in range(4)]
    enriched = await enricher.enrich_many(wines)

    assert [w.id for w in enriched] == ["HED0", "HED1", "HED2", "HED3"]
    assert all(w.style_tags == ["dry"] for w in enriched)
