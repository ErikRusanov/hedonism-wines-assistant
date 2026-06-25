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


class AnswerChunk(BaseModel):
    """One streamed slice of answer prose (an SSE token delta)."""

    delta: str


class AnswerCompletion(BaseModel):
    """Terminal stream event carrying the data known only once generation ends."""

    citations: list[WineCitation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


# What the generation stage yields: zero+ ``AnswerChunk``s then exactly one
# ``AnswerCompletion``. The I-7 SSE endpoint serialises these; the same stream
# collapses into a :class:`ChatResponse` for the non-streaming path.
type ChatStreamEvent = AnswerChunk | AnswerCompletion
