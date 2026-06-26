"""Unit tests for the chat orchestration service (I-6).

Parser, retriever and generator are replaced with fakes, so the tests pin the
control flow — which guardrail fires, what never gets called, and how the stream
collapses into a :class:`ChatResponse` — without any model or Qdrant.
"""

from __future__ import annotations

from hedonism_assistant.config import Settings
from hedonism_assistant.generation.fallbacks import (
    EMPTY_RETRIEVAL_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
)
from hedonism_assistant.generation.service import ChatService
from hedonism_assistant.models.chat import AnswerChunk, AnswerCompletion
from hedonism_assistant.models.query import ParsedQuery, PriceRange, QueryIntent, WineFilters
from hedonism_assistant.models.wine import RetrievedWine
from tests.fixtures.wines import sample_wines


class _FakeParser:
    def __init__(self, parsed: ParsedQuery) -> None:
        self._parsed = parsed

    async def parse(self, message: str) -> ParsedQuery:
        return self._parsed


class _FakeRetriever:
    def __init__(self, result: list[RetrievedWine]) -> None:
        self._result = result
        self.calls = 0

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedWine]:
        self.calls += 1
        return self._result


class _FakeGenerator:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas
        self.calls = 0

    async def stream(self, query: ParsedQuery, retrieved: list[RetrievedWine]):
        self.calls += 1
        for delta in self._deltas:
            yield delta


def _service(parser, retriever, generator) -> ChatService:
    return ChatService(parser, retriever, generator, Settings(openrouter_api_key="test"))


def _retrieved() -> list[RetrievedWine]:
    return [RetrievedWine(wine=w, score=1.0 - i * 0.1) for i, w in enumerate(sample_wines())]


async def test_out_of_scope_short_circuits() -> None:
    parsed = ParsedQuery(semantic_query="weather", intent=QueryIntent.OUT_OF_SCOPE)
    retriever = _FakeRetriever([])
    generator = _FakeGenerator(["should not run"])
    service = _service(_FakeParser(parsed), retriever, generator)

    response = await service.answer("what's the weather?")

    assert response.answer == OUT_OF_SCOPE_MESSAGE
    assert response.suggestions  # nudges offered
    assert response.citations == []
    assert retriever.calls == 0
    assert generator.calls == 0


async def test_other_drinks_redirects_to_spirits() -> None:
    parsed = ParsedQuery(semantic_query="good whisky", intent=QueryIntent.OTHER_DRINKS)
    retriever = _FakeRetriever([])
    generator = _FakeGenerator(["should not run"])
    service = _service(_FakeParser(parsed), retriever, generator)

    response = await service.answer("do you have any good whisky?")

    assert "spirits" in response.answer.lower()
    assert "hedonism.co.uk/spirits" in response.answer
    assert response.suggestions  # nudges back toward wine
    assert response.citations == []
    assert retriever.calls == 0
    assert generator.calls == 0


async def test_empty_retrieval_skips_generation() -> None:
    parsed = ParsedQuery(
        semantic_query="red Bordeaux",
        intent=QueryIntent.RECOMMENDATION,
        filters=WineFilters(price_range=PriceRange(max=20)),
    )
    retriever = _FakeRetriever([])
    generator = _FakeGenerator(["should not run"])
    service = _service(_FakeParser(parsed), retriever, generator)

    response = await service.answer("a grand cru under £20")

    assert response.answer == EMPTY_RETRIEVAL_MESSAGE
    assert any("£20" in s for s in response.suggestions)
    assert retriever.calls == 1
    assert generator.calls == 0


async def test_happy_path_assembles_answer_and_citations() -> None:
    parsed = ParsedQuery(semantic_query="red Bordeaux", intent=QueryIntent.RECOMMENDATION)
    retrieved = _retrieved()
    generator = _FakeGenerator(["The Pichon [1] is superb, ", "or the Brunello [3]."])
    service = _service(_FakeParser(parsed), _FakeRetriever(retrieved), generator)

    response = await service.answer("a red Bordeaux")

    assert response.answer == "The Pichon [1] is superb, or the Brunello [3]."
    assert [c.wine_id for c in response.citations] == [
        retrieved[0].wine.id,
        retrieved[2].wine.id,
    ]
    assert response.suggestions == []


async def test_low_confidence_attaches_disambiguation_suggestions() -> None:
    parsed = ParsedQuery(
        semantic_query="something nice",
        intent=QueryIntent.RECOMMENDATION,
        confident=False,
    )
    generator = _FakeGenerator(["The Pichon [1] is lovely."])
    service = _service(_FakeParser(parsed), _FakeRetriever(_retrieved()), generator)

    response = await service.answer("something nice")

    # Still answered from semantics, but nudges the user to narrow down.
    assert response.answer == "The Pichon [1] is lovely."
    assert response.suggestions  # disambiguation hints offered
    assert response.citations  # and citations still extracted


async def test_confident_happy_path_has_no_extra_suggestions() -> None:
    parsed = ParsedQuery(semantic_query="red Bordeaux", intent=QueryIntent.RECOMMENDATION)
    generator = _FakeGenerator(["The Pichon [1] is superb."])
    service = _service(_FakeParser(parsed), _FakeRetriever(_retrieved()), generator)

    response = await service.answer("a red Bordeaux")

    assert response.suggestions == []


async def test_stream_emits_chunks_then_one_completion() -> None:
    parsed = ParsedQuery(semantic_query="q", intent=QueryIntent.RECOMMENDATION)
    generator = _FakeGenerator(["a ", "b ", "c [1]"])
    service = _service(_FakeParser(parsed), _FakeRetriever(_retrieved()), generator)

    events = [event async for event in service.answer_stream("q")]

    assert all(isinstance(e, AnswerChunk) for e in events[:-1])
    assert isinstance(events[-1], AnswerCompletion)
    assert "".join(e.delta for e in events[:-1]) == "a b c [1]"
    assert events[-1].citations[0].wine_id == sample_wines()[0].id
