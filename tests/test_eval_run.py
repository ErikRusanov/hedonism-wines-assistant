"""Tests for the eval orchestrator with fake pipeline stages (I-8).

No network and no Qdrant: parser, retriever, generator and judge are fakes, so
the tests pin aggregation, the retrieval/guardrail split, threshold gating and
report writing — mirroring the fake-store style of ``test_index.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from hedonism_assistant.config import Settings
from hedonism_assistant.eval.golden import GoldenCase
from hedonism_assistant.eval.run import run_eval
from hedonism_assistant.models.query import ParsedQuery, QueryIntent, WineFilters
from hedonism_assistant.models.wine import RetrievedWine, WineColor
from tests.fixtures.wines import sample_wines


class _FakeParser:
    """Returns a preset :class:`ParsedQuery` keyed on the message text."""

    def __init__(self, mapping: dict[str, ParsedQuery]) -> None:
        self._mapping = mapping

    async def parse(self, message: str) -> ParsedQuery:
        return self._mapping[message]


class _FakeRetriever:
    """Returns all sample wines, or nothing for the 'impossible' query."""

    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: ParsedQuery) -> list[RetrievedWine]:
        self.calls += 1
        if query.semantic_query == "impossible":
            return []
        return [RetrievedWine(wine=w, score=1.0 - i * 0.1) for i, w in enumerate(sample_wines())]


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, query: ParsedQuery, retrieved: list[RetrievedWine]):
        self.calls += 1
        yield "The Pichon [1] is excellent."


class _FakeJudge:
    def __init__(self, faith: float | None, relevancy: float | None) -> None:
        self._faith = faith
        self._relevancy = relevancy

    async def faithfulness(self, question, answer, retrieved) -> float | None:
        return self._faith

    async def answer_relevancy(self, question, answer) -> float | None:
        return self._relevancy


def _golden() -> list[GoldenCase]:
    return [
        GoldenCase(
            id="G1",
            question="red Bordeaux",
            relevance=WineFilters(color=[WineColor.RED], region=["Bordeaux"]),
        ),
        GoldenCase(id="G2", question="whisky", expected_fallback="other_drinks"),
        GoldenCase(id="G3", question="impossible", expected_fallback="empty"),
    ]


def _parser() -> _FakeParser:
    return _FakeParser(
        {
            "red Bordeaux": ParsedQuery(
                semantic_query="red Bordeaux", intent=QueryIntent.RECOMMENDATION
            ),
            "whisky": ParsedQuery(semantic_query="whisky", intent=QueryIntent.OTHER_DRINKS),
            "impossible": ParsedQuery(
                semantic_query="impossible", intent=QueryIntent.RECOMMENDATION
            ),
        }
    )


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test",
        eval_report_path=str(tmp_path / "eval_report.json"),
        **overrides,
    )


async def test_aggregates_and_writes_report(tmp_path: Path) -> None:
    retriever, generator = _FakeRetriever(), _FakeGenerator()
    report = await run_eval(
        _settings(tmp_path),
        golden=_golden(),
        parser=_parser(),
        retriever=retriever,
        generator=generator,
        judge=_FakeJudge(1.0, 0.9),
    )

    assert report.dataset_size == 3
    assert report.evaluated == 1  # only G1 is a retrieval case (G2/G3 are guardrails)
    # HED1001 (red Bordeaux) is first and relevant -> perfect retrieval on G1.
    assert report.hit_at_1 == 1.0
    assert report.mrr == 1.0
    assert report.faithfulness == 1.0
    assert report.answer_relevancy == 0.9
    assert report.judged == 1
    # The other-drinks intent short-circuits before retrieval.
    assert retriever.calls == 2  # G1 + G3 (G2 never retrieves)
    assert generator.calls == 1  # only the happy G1 reaches generation

    written = json.loads((tmp_path / "eval_report.json").read_text())
    assert written["dataset_size"] == 3
    assert written["passed"] is True


async def test_guardrail_branches_recorded(tmp_path: Path) -> None:
    report = await run_eval(
        _settings(tmp_path),
        golden=_golden(),
        parser=_parser(),
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        judge=_FakeJudge(1.0, 1.0),
    )
    by_id = {c.id: c for c in report.cases}
    assert by_id["G2"].fallback == "other_drinks" and by_id["G2"].fallback_ok is True
    assert by_id["G3"].fallback == "empty" and by_id["G3"].fallback_ok is True
    assert report.fallback_cases == 2
    assert report.fallback_ok == 2


async def test_latency_is_populated(tmp_path: Path) -> None:
    report = await run_eval(
        _settings(tmp_path),
        golden=_golden(),
        parser=_parser(),
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        judge=_FakeJudge(1.0, 1.0),
    )
    g1 = next(c for c in report.cases if c.id == "G1")
    assert g1.latency.parse_ms >= 0.0
    assert g1.latency.generate_ms > 0.0
    assert g1.latency.total_ms > 0.0


async def test_threshold_gate_fails_on_low_faithfulness(tmp_path: Path) -> None:
    report = await run_eval(
        _settings(tmp_path),
        golden=_golden(),
        parser=_parser(),
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        judge=_FakeJudge(0.10, 1.0),  # below the 0.85 faithfulness floor
    )
    assert report.faithfulness == 0.10
    assert report.passed is False


async def test_no_judge_leaves_quality_metrics_none(tmp_path: Path) -> None:
    report = await run_eval(
        _settings(tmp_path),
        golden=_golden(),
        parser=_parser(),
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        judge_enabled=False,
    )
    assert report.faithfulness is None
    assert report.answer_relevancy is None
    assert report.judged == 0
    # Retrieval thresholds still pass, and None quality metrics are skipped.
    assert report.passed is True


async def test_low_confidence_is_recorded(tmp_path: Path) -> None:
    parser = _FakeParser(
        {
            "red Bordeaux": ParsedQuery(
                semantic_query="red Bordeaux",
                intent=QueryIntent.RECOMMENDATION,
                confident=False,
            ),
            "whisky": ParsedQuery(semantic_query="whisky", intent=QueryIntent.OTHER_DRINKS),
            "impossible": ParsedQuery(
                semantic_query="impossible", intent=QueryIntent.RECOMMENDATION
            ),
        }
    )
    report = await run_eval(
        _settings(tmp_path),
        golden=_golden(),
        parser=parser,
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
        judge=_FakeJudge(1.0, 1.0),
    )
    assert next(c for c in report.cases if c.id == "G1").low_confidence is True
