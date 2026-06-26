"""Request/response contracts for the ``POST /search`` endpoint (I-7).

``/search`` exposes the retrieval pipeline directly (parse → hybrid retrieve →
rerank) without generation. The response echoes the :class:`ParsedQuery` so a
caller can see exactly which hard filters were applied — the point of the
endpoint is to prove that "red Bordeaux under £50" really filters on
``color=red``, ``region=Bordeaux``, ``price<=50`` rather than searching as text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hedonism_assistant.models.query import ParsedQuery
from hedonism_assistant.models.wine import Wine


class SearchRequest(BaseModel):
    """A single search query (stateless, no session)."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional cap on returned hits; defaults to the configured rerank_top_k.",
    )


class SearchHit(BaseModel):
    """One retrieved wine plus its fusion/rerank scores."""

    wine: Wine
    score: float
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    """Search results plus the parsed query that produced them."""

    parsed: ParsedQuery
    hits: list[SearchHit] = Field(default_factory=list)
