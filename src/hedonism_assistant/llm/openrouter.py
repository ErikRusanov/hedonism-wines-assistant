"""Thin async wrapper over the OpenAI SDK pointed at OpenRouter.

Centralises model selection, fallback chains and retries so the rest of the
codebase never talks to the raw SDK. Both generation and embeddings go through
OpenRouter's OpenAI-compatible API, keeping the service to a single provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from functools import lru_cache
from typing import TypeVar

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.logging_config import get_logger

logger = get_logger(__name__)

# Transient OpenRouter/upstream failures worth retrying on the same model.
RETRYABLE_ERRORS = (APITimeoutError, RateLimitError, APIError)

T = TypeVar("T")


class OpenRouterClient:
    """Async client exposing the two primitives the RAG pipeline needs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # When routing through a gated reverse proxy, every request must carry the
        # proxy's auth header (the proxy then forwards to OpenRouter upstream).
        default_headers = (
            {"X-Proxy-Token": settings.proxy_auth_token} if settings.proxy_auth_token else None
        )
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout_seconds,
            max_retries=0,  # retries are owned by this wrapper, not the SDK
            default_headers=default_headers,
        )

    async def _with_retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run ``operation`` with exponential backoff on transient errors.

        Attempt count is driven by ``settings.max_retries`` so retry behaviour
        stays configurable rather than baked into a decorator.
        """
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        ):
            with attempt:
                return await operation()
        raise AssertionError("unreachable: AsyncRetrying always yields or raises")

    async def chat(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        fallback_models: Sequence[str] | None = None,
        temperature: float = 0.2,
        **kwargs: object,
    ) -> str:
        """Return a single completion, trying each model in the fallback chain.

        Each model is retried independently on transient errors; once a model is
        exhausted the next one in the chain is tried.
        """
        fallbacks = (
            fallback_models
            if fallback_models is not None
            else self._settings.generation_fallback_models
        )
        chain = [model or self._settings.generation_model, *fallbacks]
        payload = list(messages)

        last_error: Exception | None = None
        for candidate in chain:
            try:
                return await self._with_retry(
                    lambda c=candidate: self._create_chat(c, payload, temperature, **kwargs)
                )
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                logger.warning("chat_model_failed", model=candidate, error=str(exc))
        raise RuntimeError("all chat models in the fallback chain failed") from last_error

    async def _create_chat(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        **kwargs: object,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        fallback_models: Sequence[str] | None = None,
        temperature: float = 0.2,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Yield answer tokens as they arrive (used by the SSE endpoint).

        Each model in the fallback chain is tried until one *starts* streaming:
        a failure before the first token degrades to the next model, so a dead
        primary degrades gracefully instead of surfacing a 503. Once the first
        token has been yielded the stream is committed — a mid-stream failure is
        not retried and surfaces to the caller (we can't un-send tokens).
        """
        fallbacks = (
            fallback_models
            if fallback_models is not None
            else self._settings.generation_fallback_models
        )
        chain = [model or self._settings.generation_model, *fallbacks]
        payload = list(messages)

        last_error: Exception | None = None
        for candidate in chain:
            try:
                stream = await self._create_stream(candidate, payload, temperature, **kwargs)
                first = await _first_delta(stream)
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                logger.warning("chat_stream_model_failed", model=candidate, error=str(exc))
                continue
            if first is not None:
                yield first
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return
        raise RuntimeError("all chat stream models in the fallback chain failed") from last_error

    async def _create_stream(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        **kwargs: object,
    ) -> AsyncIterator[object]:
        """Open a streaming completion (its own seam so tests can stub it)."""
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts and return one vector per input, in order."""
        chosen = model or self._settings.embedding_model
        inputs = list(texts)

        async def _create() -> list[list[float]]:
            response = await self._client.embeddings.create(model=chosen, input=inputs)
            return [item.embedding for item in response.data]

        return await self._with_retry(_create)


async def _first_delta(stream: AsyncIterator[object]) -> str | None:
    """Advance ``stream`` to its first non-empty content delta, or ``None``.

    Errors raised while pulling the first token propagate to the caller's
    fallback boundary; a clean stream with no content returns ``None`` (an empty
    answer is not a failure and must not trigger a fallback).
    """
    async for chunk in stream:
        choices = getattr(chunk, "choices", None)
        delta = choices[0].delta.content if choices else None
        if delta:
            return delta
    return None


@lru_cache
def get_openrouter_client() -> OpenRouterClient:
    """Return the cached OpenRouter client built from settings."""
    return OpenRouterClient(get_settings())
