"""Unit tests for the OpenRouter client's retry and fallback behaviour.

The HTTP layer is never exercised: ``_create_chat`` is replaced with a stub so we
test only the wrapper's own logic (model chaining and configurable retries).
"""

import httpx
import pytest
from openai import APITimeoutError

from hedonism_assistant.config import Settings
from hedonism_assistant.llm.openrouter import OpenRouterClient


def _timeout() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "http://test"))


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
