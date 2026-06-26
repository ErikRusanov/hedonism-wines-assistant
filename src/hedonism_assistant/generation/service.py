"""Chat orchestration: parse → guardrails → retrieve → generate (I-6).

``ChatService`` is the in-process seam the serving layer (I-7) will sit behind. It
ties the existing stages together and owns the guardrails that must short-circuit
*before* generation:

1. **Other drinks** — if the user asks about a non-wine drink (spirits, beer, …), we
   redirect them to Hedonism's spirits range instead of guessing at wines.
2. **Out-of-scope** — if query understanding flags the message as off-domain, we
   never retrieve or call the model; we return a fixed redirect plus nudges.
3. **Empty retrieval** — if nothing matches the (filtered) query, there is nothing
   to ground on, so we return a fixed apology plus filter-relaxation suggestions.

Only the happy path reaches the generation model. The service speaks in stream
events (:data:`ChatStreamEvent`) so the SSE endpoint can forward them verbatim;
``answer`` collapses the same stream into a :class:`ChatResponse`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.generation.citations import extract_citations
from hedonism_assistant.generation.fallbacks import (
    EMPTY_RETRIEVAL_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    empty_retrieval_suggestions,
    low_confidence_suggestions,
    other_drinks_message,
    out_of_scope_suggestions,
)
from hedonism_assistant.generation.generator import AnswerGenerator, get_generator
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.chat import (
    AnswerChunk,
    AnswerCompletion,
    ChatResponse,
    ChatStreamEvent,
    WineCitation,
)
from hedonism_assistant.models.query import QueryIntent
from hedonism_assistant.retrieval.query_parser import QueryParser, get_query_parser
from hedonism_assistant.retrieval.retriever import Retriever, get_retriever

logger = get_logger(__name__)


class ChatService:
    """Answer a single stateless chat turn end to end."""

    __slots__ = ("_parser", "_retriever", "_generator", "_settings")

    def __init__(
        self,
        parser: QueryParser,
        retriever: Retriever,
        generator: AnswerGenerator,
        settings: Settings,
    ) -> None:
        self._parser = parser
        self._retriever = retriever
        self._generator = generator
        self._settings = settings

    async def answer_stream(self, message: str) -> AsyncIterator[ChatStreamEvent]:
        """Stream the answer to ``message`` as chunks then one completion event."""
        parsed = await self._parser.parse(message)

        if parsed.intent is QueryIntent.OTHER_DRINKS:
            logger.info("chat_other_drinks")
            yield AnswerChunk(delta=other_drinks_message(self._settings.spirits_url))
            yield AnswerCompletion(
                suggestions=out_of_scope_suggestions(
                    limit=self._settings.generation_max_suggestions
                )
            )
            return

        if parsed.intent is QueryIntent.OUT_OF_SCOPE:
            logger.info("chat_out_of_scope")
            yield AnswerChunk(delta=OUT_OF_SCOPE_MESSAGE)
            yield AnswerCompletion(
                suggestions=out_of_scope_suggestions(
                    limit=self._settings.generation_max_suggestions
                )
            )
            return

        retrieved = await self._retriever.retrieve(parsed)
        if not retrieved:
            logger.info("chat_empty_retrieval")
            yield AnswerChunk(delta=EMPTY_RETRIEVAL_MESSAGE)
            yield AnswerCompletion(
                suggestions=empty_retrieval_suggestions(
                    parsed.filters, limit=self._settings.generation_max_suggestions
                )
            )
            return

        parts: list[str] = []
        async for delta in self._generator.stream(parsed, retrieved):
            parts.append(delta)
            yield AnswerChunk(delta=delta)

        # When query understanding was unsure (a parse failure or ambiguous ask),
        # we still answered from pure semantics, but steer the user to disambiguate
        # so the next turn can filter precisely.
        suggestions: list[str] = []
        if not parsed.confident:
            logger.info("chat_low_confidence")
            suggestions = low_confidence_suggestions(
                limit=self._settings.generation_max_suggestions
            )

        answer = "".join(parts)
        yield AnswerCompletion(
            citations=extract_citations(answer, retrieved), suggestions=suggestions
        )

    async def answer(self, message: str) -> ChatResponse:
        """Collect the stream into a non-streaming :class:`ChatResponse`."""
        parts: list[str] = []
        citations: list[WineCitation] = []
        suggestions: list[str] = []
        async for event in self.answer_stream(message):
            match event:
                case AnswerChunk(delta=delta):
                    parts.append(delta)
                case AnswerCompletion(citations=cited, suggestions=hints):
                    citations = cited
                    suggestions = hints
        return ChatResponse(answer="".join(parts), citations=citations, suggestions=suggestions)


@lru_cache
def get_chat_service() -> ChatService:
    """Return the cached chat service wired to the shared pipeline singletons."""
    return ChatService(
        parser=get_query_parser(),
        retriever=get_retriever(),
        generator=get_generator(),
        settings=get_settings(),
    )
