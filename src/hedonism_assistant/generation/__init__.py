"""Answer generation (I-6): grounded, cited, streamed responses over retrieval.

The package turns reranked :class:`RetrievedWine` cards into a streamed answer with
citations and guardrails. ``ChatService`` is the orchestration seam the serving
layer (I-7) builds its endpoints on.
"""

from __future__ import annotations

from hedonism_assistant.generation.generator import AnswerGenerator, get_generator
from hedonism_assistant.generation.service import ChatService, get_chat_service

__all__ = [
    "AnswerGenerator",
    "ChatService",
    "get_chat_service",
    "get_generator",
]
