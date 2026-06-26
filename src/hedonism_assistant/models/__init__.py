"""Pydantic domain contracts shared across the scrape, index and serve tracks."""

from hedonism_assistant.models.chat import ChatRequest, ChatResponse, WineCitation
from hedonism_assistant.models.query import (
    ParsedQuery,
    PriceRange,
    QueryIntent,
    VintageRange,
    WineFilters,
)
from hedonism_assistant.models.search import SearchHit, SearchRequest, SearchResponse
from hedonism_assistant.models.wine import (
    Availability,
    CriticScore,
    RetrievedWine,
    Wine,
    WineCategory,
    WineColor,
)

__all__ = [
    "Wine",
    "WineCategory",
    "WineColor",
    "Availability",
    "CriticScore",
    "RetrievedWine",
    "ParsedQuery",
    "QueryIntent",
    "WineFilters",
    "PriceRange",
    "VintageRange",
    "ChatRequest",
    "ChatResponse",
    "WineCitation",
    "SearchRequest",
    "SearchResponse",
    "SearchHit",
]
