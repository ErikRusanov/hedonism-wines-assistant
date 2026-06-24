"""Request/response contracts for the chat and search endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class ChatRequest(BaseModel):
    """A single, stateless chat turn (no session/history is persisted)."""

    message: str = Field(min_length=1, max_length=2000)


class WineCitation(BaseModel):
    """A grounded reference to a wine used in the answer."""

    wine_id: str
    name: str
    url: HttpUrl
    price: float | None = None
    currency: str = "GBP"


class ChatResponse(BaseModel):
    """Non-streaming representation of an answer (the SSE stream emits the same data).

    The streaming endpoint sends ``answer`` incrementally and the citations and
    suggestions once retrieval/generation completes.
    """

    answer: str
    citations: list[WineCitation] = Field(default_factory=list)
    suggestions: list[str] = Field(
        default_factory=list,
        description="Follow-up clarifications offered when retrieval is weak or empty.",
    )
