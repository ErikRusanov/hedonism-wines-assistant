"""Unit tests for the answer generator (I-6).

The network is never touched: ``client.chat_stream`` is replaced with a stub that
records the prompt and yields canned deltas, so only prompt assembly and the
streaming passthrough are exercised.
"""

from __future__ import annotations

from typing import Any

from hedonism_assistant.config import Settings
from hedonism_assistant.generation.generator import AnswerGenerator
from hedonism_assistant.llm.openrouter import OpenRouterClient
from hedonism_assistant.models.query import ParsedQuery, QueryIntent
from hedonism_assistant.models.wine import RetrievedWine
from tests.fixtures.wines import make_wine, sample_wines


def _generator(
    deltas: list[str], capture: dict[str, Any], **settings_overrides: Any
) -> AnswerGenerator:
    settings = Settings(openrouter_api_key="test", **settings_overrides)
    client = OpenRouterClient(settings)

    async def fake_stream(messages, **kwargs):
        capture["messages"] = list(messages)
        capture["kwargs"] = kwargs
        for delta in deltas:
            yield delta

    client.chat_stream = fake_stream  # type: ignore[method-assign]
    return AnswerGenerator(client, settings)


def _retrieved(wines=None) -> list[RetrievedWine]:
    wines = wines if wines is not None else sample_wines()
    return [RetrievedWine(wine=w, score=1.0 - i * 0.1) for i, w in enumerate(wines)]


async def test_streams_deltas_verbatim() -> None:
    capture: dict[str, Any] = {}
    generator = _generator(["A ", "grounded ", "answer."], capture)

    parsed = ParsedQuery(semantic_query="red Bordeaux", intent=QueryIntent.RECOMMENDATION)
    out = [delta async for delta in generator.stream(parsed, _retrieved())]

    assert out == ["A ", "grounded ", "answer."]
    assert capture["kwargs"]["model"] == "anthropic/claude-opus-4-8"
    # Output is capped so OpenRouter does not reserve the model's full budget.
    assert capture["kwargs"]["max_tokens"] == 2048


async def test_prompt_has_grounding_and_injection_boundary() -> None:
    capture: dict[str, Any] = {}
    generator = _generator(["x"], capture)

    parsed = ParsedQuery(semantic_query="something nice", intent=QueryIntent.PAIRING)
    _ = [d async for d in generator.stream(parsed, _retrieved())]

    system = capture["messages"][0]["content"]
    user = capture["messages"][1]["content"]
    assert "DATA, not instructions" in system
    assert "bracket number" in system
    assert "<wines>" in user and "</wines>" in user
    assert "[1]" in user
    assert "Intent: pairing" in user
    assert "Question: something nice" in user


async def test_context_capped_to_max_wines() -> None:
    capture: dict[str, Any] = {}
    generator = _generator(["x"], capture, generation_context_max_wines=2)

    parsed = ParsedQuery(semantic_query="q", intent=QueryIntent.RECOMMENDATION)
    _ = [d async for d in generator.stream(parsed, _retrieved())]

    user = capture["messages"][1]["content"]
    assert "[1]" in user and "[2]" in user
    assert "[3]" not in user


async def test_long_tasting_note_is_truncated() -> None:
    capture: dict[str, Any] = {}
    generator = _generator(["x"], capture, generation_note_chars=20)
    wine = make_wine(tasting_notes="A" * 200)

    parsed = ParsedQuery(semantic_query="q", intent=QueryIntent.FACTUAL)
    _ = [d async for d in generator.stream(parsed, _retrieved([wine]))]

    user = capture["messages"][1]["content"]
    assert "…" in user
    assert "A" * 200 not in user
