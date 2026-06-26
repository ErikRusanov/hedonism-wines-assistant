"""Endpoint tests for ``POST /chat`` (I-7).

The chat service is replaced via ``app.dependency_overrides`` so the SSE adapter
is exercised — frame serialisation, guardrail passthrough, validation and the
upstream-failure mapping — without any model or Qdrant.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hedonism_assistant.api.app import create_app
from hedonism_assistant.config import Settings
from hedonism_assistant.generation.fallbacks import OUT_OF_SCOPE_MESSAGE
from hedonism_assistant.generation.service import get_chat_service
from hedonism_assistant.models.chat import AnswerChunk, AnswerCompletion, WineCitation


class _FakeService:
    """Stand-in chat service whose ``answer_stream`` yields scripted events."""

    def __init__(self, events) -> None:
        self._events = events

    async def answer_stream(self, message: str):
        for event in self._events:
            if isinstance(event, Exception):
                raise event
            yield event


def _client(service: _FakeService) -> TestClient:
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(app)


def _data_frames(body: str) -> list[dict]:
    """Parse the JSON payloads of ``data:`` frames, skipping the done marker."""
    out: list[dict] = []
    for frame in body.split("\n\n"):
        if not frame.strip() or frame.startswith("event: done"):
            continue
        line = next((ln for ln in frame.splitlines() if ln.startswith("data: ")), None)
        if line and line[6:] != "{}":
            out.append(json.loads(line[6:]))
    return out


def test_chat_streams_chunks_then_completion() -> None:
    citation = WineCitation(wine_id="HED1", name="Pichon", url="https://hedonism.co.uk/p/1")
    events = [
        AnswerChunk(delta="The Pichon "),
        AnswerChunk(delta="[1] is superb."),
        AnswerCompletion(citations=[citation], suggestions=[]),
    ]
    response = _client(_FakeService(events)).post("/chat", json={"message": "a red Bordeaux"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _data_frames(response.text)
    answer = "".join(f["delta"] for f in frames if "delta" in f)
    assert answer == "The Pichon [1] is superb."
    completion = next(f for f in frames if "citations" in f)
    assert completion["citations"][0]["wine_id"] == "HED1"


def test_chat_forwards_out_of_scope_guardrail() -> None:
    events = [
        AnswerChunk(delta=OUT_OF_SCOPE_MESSAGE),
        AnswerCompletion(suggestions=["Try asking about a wine region"]),
    ]
    response = _client(_FakeService(events)).post("/chat", json={"message": "weather?"})

    frames = _data_frames(response.text)
    assert frames[0]["delta"] == OUT_OF_SCOPE_MESSAGE
    assert frames[-1]["suggestions"] == ["Try asking about a wine region"]


def test_chat_rejects_empty_message() -> None:
    response = _client(_FakeService([])).post("/chat", json={"message": ""})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_chat_maps_upstream_failure_to_503() -> None:
    service = _FakeService([RuntimeError("all chat models in the fallback chain failed")])
    response = _client(service).post("/chat", json={"message": "a red Bordeaux"})
    assert response.status_code == 503
    assert response.json()["error"] == "upstream_unavailable"
