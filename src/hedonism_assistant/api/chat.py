"""``POST /chat`` — grounded answer streamed over Server-Sent Events (I-7).

The orchestration lives in :class:`ChatService` (I-6); this endpoint is a thin
SSE adapter that forwards its :data:`ChatStreamEvent`s verbatim as ``data:``
frames. We pull the *first* event before returning the streaming response so a
pre-stream failure (an exhausted OpenRouter fallback chain raises
:class:`RuntimeError`) surfaces as a clean 503 instead of a half-open stream —
once the 200 response body has started there is no way to change the status.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from hedonism_assistant.generation.service import ChatService, get_chat_service
from hedonism_assistant.models.chat import ChatRequest, ChatStreamEvent

router = APIRouter(tags=["chat"])

# Disable proxy/browser buffering so tokens reach the client as they stream.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _frame(event: ChatStreamEvent) -> str:
    """Serialise one stream event as an SSE ``data:`` frame."""
    return f"data: {event.model_dump_json()}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    events = service.answer_stream(request.message).__aiter__()
    # Surface a pre-stream error here (caught by the 503 handler); StopAsyncIteration
    # would only happen if the service yielded nothing, which it never does.
    first = await events.__anext__()

    async def body() -> AsyncIterator[str]:
        yield _frame(first)
        async for event in events:
            yield _frame(event)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream", headers=_SSE_HEADERS)
