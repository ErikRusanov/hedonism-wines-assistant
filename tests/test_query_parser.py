"""Unit tests for the query parser.

The network is never touched: ``client.chat`` is replaced with a stub returning
a JSON string, so only the parser's own coercion and fallback logic is tested.
"""

import json

from hedonism_assistant.config import Settings
from hedonism_assistant.llm.openrouter import OpenRouterClient
from hedonism_assistant.models.query import QueryIntent
from hedonism_assistant.models.wine import WineColor
from hedonism_assistant.retrieval.query_parser import QueryParser
from hedonism_assistant.retrieval.taxonomy import Taxonomy


def _client() -> OpenRouterClient:
    return OpenRouterClient(Settings(openrouter_api_key="test"))


def _parser_returning(payload: object, *, taxonomy: Taxonomy | None = None) -> QueryParser:
    client = _client()

    async def fake_chat(messages, **kwargs) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload)

    client.chat = fake_chat  # type: ignore[method-assign]
    return QueryParser(client, Settings(openrouter_api_key="test"), taxonomy=taxonomy)


async def test_happy_path_extracts_filters() -> None:
    parser = _parser_returning(
        {
            "semantic_query": "red Bordeaux",
            "intent": "recommendation",
            "filters": {
                "color": ["red"],
                "country": ["France"],
                "region": ["Bordeaux"],
                "price_range": {"max": 50},
            },
        }
    )

    parsed = await parser.parse("red Bordeaux under £50")

    assert parsed.confident is True
    assert parsed.semantic_query == "red Bordeaux"
    assert parsed.intent is QueryIntent.RECOMMENDATION
    assert parsed.filters.color == [WineColor.RED]
    assert parsed.filters.country == ["France"]
    assert parsed.filters.region == ["Bordeaux"]
    assert parsed.filters.price_range is not None
    assert parsed.filters.price_range.max == 50
    assert parsed.filters.price_range.min is None


async def test_unknown_region_is_dropped_against_taxonomy() -> None:
    parser = _parser_returning(
        {
            "semantic_query": "something",
            "intent": "recommendation",
            "filters": {"region": ["Bordeaux", "Atlantis"]},
        },
        taxonomy=Taxonomy(regions=frozenset({"Bordeaux"})),
    )

    parsed = await parser.parse("wines from Bordeaux and Atlantis")

    assert parsed.filters.region == ["Bordeaux"]


async def test_producer_is_validated_against_taxonomy() -> None:
    # Regression: a producer query must become a hard "producer" filter, matched
    # diacritic-insensitively against the catalogue ("Dom Pérignon" -> "Dom Perignon").
    parser = _parser_returning(
        {
            "semantic_query": "Dom Pérignon champagne",
            "intent": "factual",
            "filters": {"producer": ["Dom Pérignon"]},
        },
        taxonomy=Taxonomy(producers=frozenset({"Dom Perignon"})),
    )

    parsed = await parser.parse("tell me about Dom Pérignon")

    assert parsed.filters.producer == ["Dom Perignon"]
    assert parsed.intent is QueryIntent.FACTUAL


async def test_unknown_producer_is_dropped() -> None:
    parser = _parser_returning(
        {
            "semantic_query": "x",
            "intent": "factual",
            "filters": {"producer": ["Chateau Imaginaire"]},
        },
        taxonomy=Taxonomy(producers=frozenset({"Dom Perignon"})),
    )

    parsed = await parser.parse("anything from Chateau Imaginaire?")

    assert parsed.filters.producer == []


async def test_broken_json_falls_back_to_pure_semantic() -> None:
    parser = _parser_returning("not json at all {")

    parsed = await parser.parse("red Bordeaux under £50")

    assert parsed.confident is False
    assert parsed.semantic_query == "red Bordeaux under £50"
    assert parsed.filters.region == []
    assert parsed.intent is QueryIntent.RECOMMENDATION


async def test_model_chain_failure_falls_back_to_pure_semantic() -> None:
    client = _client()

    async def failing_chat(messages, **kwargs) -> str:
        raise RuntimeError("all chat models in the fallback chain failed")

    client.chat = failing_chat  # type: ignore[method-assign]
    parser = QueryParser(client, Settings(openrouter_api_key="test"))

    parsed = await parser.parse("red Bordeaux")

    assert parsed.confident is False
    assert parsed.semantic_query == "red Bordeaux"


async def test_parsing_disabled_skips_the_model() -> None:
    client = _client()

    async def boom(messages, **kwargs) -> str:
        raise AssertionError("chat must not be called when parsing is disabled")

    client.chat = boom  # type: ignore[method-assign]
    settings = Settings(openrouter_api_key="test", query_parsing_enabled=False)
    parser = QueryParser(client, settings)

    parsed = await parser.parse("red Bordeaux under £50")

    assert parsed.confident is True
    assert parsed.semantic_query == "red Bordeaux under £50"
    assert parsed.filters.color == []
    assert parsed.filters.price_range is None


async def test_out_of_scope_intent_is_propagated() -> None:
    parser = _parser_returning(
        {
            "semantic_query": "what's the weather like today",
            "intent": "out_of_scope",
            "filters": {},
        }
    )

    parsed = await parser.parse("what's the weather like today?")

    assert parsed.intent is QueryIntent.OUT_OF_SCOPE
    assert parsed.filters.color == []
    assert parsed.filters.region == []
    assert parsed.confident is True


async def test_other_drinks_intent_is_propagated() -> None:
    parser = _parser_returning(
        {"semantic_query": "good whisky", "intent": "other_drinks", "filters": {}}
    )

    parsed = await parser.parse("do you have any good whisky?")

    assert parsed.intent is QueryIntent.OTHER_DRINKS
    assert parsed.filters.region == []
    assert parsed.confident is True


async def test_invalid_intent_defaults_to_recommendation() -> None:
    parser = _parser_returning({"semantic_query": "x", "intent": "nonsense", "filters": {}})

    parsed = await parser.parse("x")

    assert parsed.intent is QueryIntent.RECOMMENDATION


async def test_bad_filter_value_does_not_sink_the_parse() -> None:
    parser = _parser_returning(
        {
            "semantic_query": "x",
            "intent": "recommendation",
            "filters": {
                "color": ["red", "ultraviolet"],
                "bottle_size_ml": "not-a-number",
                "in_bond": "yes",
                "min_critic_score": 92,
            },
        }
    )

    parsed = await parser.parse("x")

    assert parsed.filters.color == [WineColor.RED]
    assert parsed.filters.bottle_size_ml is None
    assert parsed.filters.in_bond is None
    assert parsed.filters.min_critic_score == 92.0
