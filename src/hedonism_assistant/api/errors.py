"""Unified error responses for the API (I-7).

Every error leaves the service in one shape — ``{"error": <code>, "detail": ...}``
— so clients (and the chat page) can handle failures uniformly. Three handlers
cover the failure modes the pipeline actually produces:

* request validation (FastAPI's :class:`RequestValidationError`) → 422;
* an exhausted OpenRouter fallback chain (raised as :class:`RuntimeError` by
  ``OpenRouterClient``) → 503, the one expected upstream failure;
* anything else → 500, without leaking internals into the body.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hedonism_assistant.logging_config import get_logger

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    """The single error envelope returned by every failing endpoint."""

    error: str
    detail: str | None = None


def _json(status_code: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail).model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach the shared exception handlers to ``app``."""

    @app.exception_handler(RequestValidationError)
    async def _on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_request", str(exc))

    @app.exception_handler(RuntimeError)
    async def _on_runtime(request: Request, exc: RuntimeError) -> JSONResponse:
        logger.error("upstream_failed", error=str(exc), path=request.url.path)
        return _json(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "upstream_unavailable",
            "The language model is temporarily unavailable. Please try again.",
        )

    @app.exception_handler(Exception)
    async def _on_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("internal_error", error=str(exc), path=request.url.path)
        return _json(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error")
