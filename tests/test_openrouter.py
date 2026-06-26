"""Unit tests for the OpenRouter client's retry and fallback behaviour.

The HTTP layer is never exercised: ``_create_chat`` is replaced with a stub so we
test only the wrapper's own logic (model chaining and configurable retries).
"""

from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from hedonism_assistant.config import Settings
from hedonism_assistant.llm.openrouter import OpenRouterClient


def _timeout() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "http://test"))


def _chunk(content: str) -> SimpleNamespace:
    """A minimal streaming chunk shaped like the OpenAI SDK's."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


async def _stream_of(contents: list[str]) -> AsyncIterator[SimpleNamespace]:
    for content in contents:
        yield _chunk(content)


async def _stream_then_raise(contents: list[str], exc: Exception) -> AsyncIterator[SimpleNamespace]:
    for content in contents:
        yield _chunk(content)
    raise exc


async def test_chat_falls_back_to_next_model() -> None:
    settings = Settings(
        openrouter_api_key="test",
        generation_model="primary",
        generation_fallback_models=["backup"],
        max_retries=1,
    )
    client = OpenRouterClient(settings)
    tried: list[str] = []

    async def fake_create(model, messages, temperature, **kwargs) -> str:
        tried.append(model)
        if model == "primary":
            raise _timeout()
        return "answer"

    client._create_chat = fake_create  # type: ignore[method-assign]

    result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "answer"
    assert tried == ["primary", "backup"]


async def test_chat_retries_same_model_per_max_retries() -> None:
    settings = Settings(openrouter_api_key="test", generation_model="only", max_retries=3)
    client = OpenRouterClient(settings)
    attempts: list[str] = []

    async def always_timeout(model, messages, temperature, **kwargs) -> str:
        attempts.append(model)
        raise _timeout()

    client._create_chat = always_timeout  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="all chat models"):
        await client.chat([{"role": "user", "content": "hi"}])

    assert len(attempts) == settings.max_retries


async def test_chat_stream_falls_back_before_first_token() -> None:
    settings = Settings(
        openrouter_api_key="test",
        generation_model="primary",
        generation_fallback_models=["backup"],
        max_retries=1,
    )
    client = OpenRouterClient(settings)
    tried: list[str] = []

    async def fake_create_stream(model, messages, temperature, **kwargs):
        tried.append(model)
        if model == "primary":
            raise _timeout()
        return _stream_of(["Hello ", "world"])

    client._create_stream = fake_create_stream  # type: ignore[method-assign]

    out = [delta async for delta in client.chat_stream([{"role": "user", "content": "hi"}])]

    assert out == ["Hello ", "world"]
    assert tried == ["primary", "backup"]


async def test_chat_stream_failure_after_first_token_surfaces() -> None:
    settings = Settings(openrouter_api_key="test", generation_model="only")
    client = OpenRouterClient(settings)

    async def fake_create_stream(model, messages, temperature, **kwargs):
        return _stream_then_raise(["Hello "], _timeout())

    client._create_stream = fake_create_stream  # type: ignore[method-assign]

    out: list[str] = []
    with pytest.raises(APITimeoutError):
        async for delta in client.chat_stream([{"role": "user", "content": "hi"}]):
            out.append(delta)

    # Committed once the first token is sent: no fallback, the error surfaces.
    assert out == ["Hello "]


async def test_chat_stream_all_models_fail_raises() -> None:
    settings = Settings(
        openrouter_api_key="test",
        generation_model="primary",
        generation_fallback_models=["backup"],
        max_retries=1,
    )
    client = OpenRouterClient(settings)

    async def always_fail(model, messages, temperature, **kwargs):
        raise _timeout()

    client._create_stream = always_fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="all chat stream models"):
        async for _ in client.chat_stream([{"role": "user", "content": "hi"}]):
            pass
