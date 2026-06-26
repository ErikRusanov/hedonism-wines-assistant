"""Endpoint tests for ``POST /search`` (I-7).

Parser and retriever are overridden via ``app.dependency_overrides`` so the
endpoint's wiring — parse echo, hit projection and the optional limit — is
pinned without a model or Qdrant.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hedonism_assistant.api.app import create_app
from hedonism_assistant.config import Settings
from hedonism_assistant.models.query import ParsedQuery, PriceRange, QueryIntent, WineFilters
from hedonism_assistant.models.wine import RetrievedWine, WineColor
from hedonism_assistant.retrieval.query_parser import get_query_parser
from hedonism_assistant.retrieval.retriever import get_retriever
from tests.fixtures.wines import sample_wines


class _FakeParser:
    def __init__(self, parsed: ParsedQuery) -> None:
        self._parsed = parsed

    async def parse(self, message: str) -> ParsedQuery:
        return self._parsed


class _FakeRetriever:
    def __init__(self, result: list[RetrievedWine]) -> None:
        self._result = result

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedWine]:
        return self._result


def _client(parsed: ParsedQuery, retrieved: list[RetrievedWine]) -> TestClient:
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_query_parser] = lambda: _FakeParser(parsed)
    app.dependency_overrides[get_retriever] = lambda: _FakeRetriever(retrieved)
    return TestClient(app)


def _retrieved() -> list[RetrievedWine]:
    return [RetrievedWine(wine=w, score=1.0 - i * 0.1) for i, w in enumerate(sample_wines())]


def test_search_echoes_parsed_filters_and_returns_hits() -> None:
    parsed = ParsedQuery(
        semantic_query="red Bordeaux",
        intent=QueryIntent.RECOMMENDATION,
        filters=WineFilters(
            color=[WineColor.RED], region=["Bordeaux"], price_range=PriceRange(max=50)
        ),
    )
    response = _client(parsed, _retrieved()).post(
        "/search", json={"query": "red Bordeaux under £50"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["parsed"]["filters"]["region"] == ["Bordeaux"]
    assert body["parsed"]["filters"]["price_range"]["max"] == 50
    assert body["parsed"]["filters"]["color"] == ["red"]
    assert len(body["hits"]) == len(sample_wines())
    assert body["hits"][0]["wine"]["id"] == sample_wines()[0].id
    assert "score" in body["hits"][0]


def test_search_applies_limit() -> None:
    parsed = ParsedQuery(semantic_query="anything", intent=QueryIntent.RECOMMENDATION)
    response = _client(parsed, _retrieved()).post("/search", json={"query": "anything", "limit": 2})

    assert response.status_code == 200
    assert len(response.json()["hits"]) == 2


def test_search_rejects_empty_query() -> None:
    parsed = ParsedQuery(semantic_query="x", intent=QueryIntent.RECOMMENDATION)
    response = _client(parsed, []).post("/search", json={"query": ""})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
