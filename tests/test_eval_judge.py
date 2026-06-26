"""Tests for the LLM-as-judge metrics (I-8).

The network is never touched: ``client.chat`` is stubbed to return canned JSON
(or raise), so only the judge's parsing, scoring and resilience are tested.
"""

from __future__ import annotations

import json

from hedonism_assistant.config import Settings
from hedonism_assistant.eval.judge import LLMJudge
from hedonism_assistant.llm.openrouter import OpenRouterClient
from hedonism_assistant.models.wine import RetrievedWine
from tests.fixtures.wines import sample_wines


def _judge_returning(payload: object) -> LLMJudge:
    settings = Settings(openrouter_api_key="test")
    client = OpenRouterClient(settings)

    async def fake_chat(messages, **kwargs) -> str:
        if isinstance(payload, Exception):
            raise payload
        return payload if isinstance(payload, str) else json.dumps(payload)

    client.chat = fake_chat  # type: ignore[method-assign]
    return LLMJudge(client, settings)


def _retrieved() -> list[RetrievedWine]:
    return [RetrievedWine(wine=w, score=1.0) for w in sample_wines()[:3]]


async def test_faithfulness_is_supported_fraction() -> None:
    judge = _judge_returning(
        {"claims": [{"text": "a", "supported": True}, {"text": "b", "supported": False}]}
    )
    score = await judge.faithfulness("q", "answer", _retrieved())
    assert score == 0.5


async def test_faithfulness_no_claims_is_vacuously_faithful() -> None:
    judge = _judge_returning({"claims": []})
    assert await judge.faithfulness("q", "answer", _retrieved()) == 1.0


async def test_answer_relevancy_clamps_to_unit() -> None:
    assert await _judge_returning({"score": 0.8}).answer_relevancy("q", "a") == 0.8
    assert await _judge_returning({"score": 1.5}).answer_relevancy("q", "a") == 1.0
    assert await _judge_returning({"score": -2}).answer_relevancy("q", "a") == 0.0


async def test_answer_relevancy_rejects_non_numeric() -> None:
    assert await _judge_returning({"score": "high"}).answer_relevancy("q", "a") is None
    assert await _judge_returning({"score": True}).answer_relevancy("q", "a") is None


async def test_metrics_degrade_to_none_on_chain_failure() -> None:
    judge = _judge_returning(RuntimeError("all chat models failed"))
    assert await judge.faithfulness("q", "a", _retrieved()) is None
    assert await judge.answer_relevancy("q", "a") is None


async def test_metrics_degrade_to_none_on_bad_json() -> None:
    judge = _judge_returning("not json at all")
    assert await judge.faithfulness("q", "a", _retrieved()) is None
    assert await judge.answer_relevancy("q", "a") is None
