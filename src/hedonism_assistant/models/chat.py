"""Request/response contracts for the chat and search endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from hedonism_assistant.models.wine import WineColor

# How many prior turns the client may replay for context. The service is still
# stateless; the client carries recent turns and we cap how far back we look.
HISTORY_MAX_TURNS = 6


class ChatTurn(BaseModel):
    """One prior message replayed by the client to give the turn context."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """A single chat turn. The service is stateless; ``history`` is client-supplied
    context for the current ``message`` (reference resolution, continuity), not a
    persisted session."""

    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)


class WineCitation(BaseModel):
    """A grounded reference to a wine used in the answer.

    Widened beyond the bare link so the UI can render a product card: the extra
    fields are all optional/defaulted, so a sparsely populated card still cites
    cleanly. ``image_path`` is a *relative* first-party path served by the app
    (``/bottles/<id>.jpg``), not the catalogue CDN URL (which Cloudflare blocks)."""

    wine_id: str
    name: str
    url: HttpUrl
    price: float | None = None
    currency: str = "GBP"

    producer: str | None = None
    region: str | None = None
    vintage: int | None = None
    color: WineColor | None = None
    grapes: list[str] = Field(default_factory=list)

    image_path: str | None = Field(
        default=None, description="First-party image path, e.g. '/bottles/HED28846.jpg'."
    )
    top_critic: str | None = Field(default=None, description="Highest-scoring critic's name.")
    top_critic_score: float | None = Field(
        default=None, description="That critic's score normalised to a 100-pt scale."
    )

    on_sale: bool = False
    sale_was_price: float | None = None


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


class QueryUnderstanding(BaseModel):
    """Stream event emitted right after query parsing, before retrieval.

    Carries a human-readable summary of what the assistant understood so the UI can
    show it as read-only chips while the answer is still being prepared.
    """

    event: Literal["understanding"] = "understanding"
    intent: str
    chips: list[str] = Field(default_factory=list)


class AnswerChunk(BaseModel):
    """One streamed slice of answer prose (an SSE token delta)."""

    event: Literal["chunk"] = "chunk"
    delta: str


class AnswerCompletion(BaseModel):
    """Terminal stream event carrying the data known only once generation ends."""

    event: Literal["completion"] = "completion"
    citations: list[WineCitation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


# What the generation stage yields: one ``QueryUnderstanding``, then zero+
# ``AnswerChunk``s, then exactly one ``AnswerCompletion``. The I-7 SSE endpoint
# serialises these (the ``event`` field is the client's discriminator); the same
# stream collapses into a :class:`ChatResponse` for the non-streaming path.
type ChatStreamEvent = QueryUnderstanding | AnswerChunk | AnswerCompletion
