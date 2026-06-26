"""Golden-set regression harness — orchestrator + CLI (I-8).

Runs the live serving pipeline (parse → retrieve → generate) over
``data/golden_set.jsonl`` and scores each case: retrieval metrics (hit@k, MRR)
on the reranked order, plus answer quality via the LLM judge (faithfulness,
answer relevancy). It composes the same pipeline singletons ``ChatService`` wires,
but does so itself so it can time each stage and capture the intermediate
``parsed``/``retrieved`` artefacts the metrics need.

Mirrors ``data/index.py``: injectable dependencies default to ``None`` and fall
back to the shared singletons; a ``@dataclass`` report is serialised to
``data/eval_report.json``; the CLI exits non-zero when a threshold is missed.

Run it as a module::

    python -m hedonism_assistant.eval.run --log-console
    python -m hedonism_assistant.eval.run --no-judge     # retrieval-only, network-light
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.eval.golden import GoldenCase, load_golden, relevant_ids
from hedonism_assistant.eval.judge import LLMJudge, get_judge
from hedonism_assistant.eval.report import (
    CaseResult,
    EvalReport,
    evaluate_thresholds,
    percentile,
    thresholds_of,
)
from hedonism_assistant.eval.retrieval_metrics import hit_at_k, mean, reciprocal_rank
from hedonism_assistant.logging_config import configure_logging, get_logger
from hedonism_assistant.models.query import ParsedQuery, QueryIntent
from hedonism_assistant.models.wine import RetrievedWine

logger = get_logger(__name__)


class _Parser(Protocol):
    async def parse(self, message: str) -> ParsedQuery: ...


class _Retriever(Protocol):
    async def retrieve(self, query: ParsedQuery) -> list[RetrievedWine]: ...


class _Generator(Protocol):
    def stream(self, query: ParsedQuery, retrieved: list[RetrievedWine]) -> AsyncIterator[str]: ...


# Maps a short-circuiting intent to the fallback label the report records. Empty
# retrieval is detected separately (no wines to ground on).
_INTENT_FALLBACK: dict[QueryIntent, str] = {
    QueryIntent.OTHER_DRINKS: "other_drinks",
    QueryIntent.OUT_OF_SCOPE: "out_of_scope",
}


async def _evaluate_case(
    case: GoldenCase,
    *,
    parser: _Parser,
    retriever: _Retriever,
    generator: _Generator,
    judge: LLMJudge | None,
    settings: Settings,
) -> CaseResult:
    """Run one case end to end, timing each stage and scoring the outcome.

    Single exit: stage timings accumulate into ``result.latency`` in place, and
    the total is summed once at the end so every branch — guardrail, empty,
    happy, or error — reports a consistent latency record.
    """
    result = CaseResult(
        id=case.id, question=case.question, expected_fallback=case.expected_fallback
    )
    latency = result.latency
    try:
        t0 = time.perf_counter()
        parsed = await parser.parse(case.question)
        latency.parse_ms = _ms_since(t0)
        result.intent = str(parsed.intent)
        result.low_confidence = not parsed.confident

        # Guardrail short-circuits, mirroring ChatService: a non-wine drink or an
        # off-domain query never retrieves or generates.
        if (fallback := _INTENT_FALLBACK.get(parsed.intent)) is not None:
            _record_fallback(result, case, fallback)
        else:
            t1 = time.perf_counter()
            retrieved = await retriever.retrieve(parsed)
            latency.retrieve_ms = _ms_since(t1)
            result.retrieved_ids = [r.wine.id for r in retrieved]

            if not retrieved:
                _record_fallback(result, case, "empty")
            else:
                _score_retrieval(result, case, retrieved, settings)
                t2 = time.perf_counter()
                answer = await _collect(generator.stream(parsed, retrieved))
                latency.generate_ms = _ms_since(t2)
                # A retrieval case that expected a fallback but generated missed.
                if case.expected_fallback is not None:
                    result.fallback_ok = False
                if judge is not None:
                    result.faithfulness = await judge.faithfulness(case.question, answer, retrieved)
                    result.answer_relevancy = await judge.answer_relevancy(case.question, answer)
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort the harness
        logger.warning("eval_case_failed", case=case.id, error=str(exc))
        result.error = str(exc)
    latency.total_ms = latency.parse_ms + latency.retrieve_ms + latency.generate_ms
    return result


def _score_retrieval(
    result: CaseResult, case: GoldenCase, retrieved: list[RetrievedWine], settings: Settings
) -> None:
    """Fill hit@k / reciprocal rank against the case's relevant ids."""
    relevant = relevant_ids(case, [r.wine for r in retrieved])
    result.relevant_ids = sorted(relevant)
    ranked = result.retrieved_ids
    result.hit_at_1 = hit_at_k(ranked, relevant, 1)
    result.hit_at_3 = hit_at_k(ranked, relevant, 3)
    result.hit_at_5 = hit_at_k(ranked, relevant, 5)
    result.hit_at_k = hit_at_k(ranked, relevant, settings.rerank_top_k)
    result.reciprocal_rank = reciprocal_rank(ranked, relevant)


def _record_fallback(result: CaseResult, case: GoldenCase, fallback: str) -> None:
    """Note which guardrail branch fired and whether the case expected it."""
    result.fallback = fallback
    result.fallback_ok = case.expected_fallback == fallback


async def run_eval(
    settings: Settings,
    *,
    limit: int | None = None,
    judge_enabled: bool | None = None,
    golden: list[GoldenCase] | None = None,
    parser: _Parser | None = None,
    retriever: _Retriever | None = None,
    generator: _Generator | None = None,
    judge: LLMJudge | None = None,
) -> EvalReport:
    """Evaluate every golden case; write the report file and return it."""
    cases = golden if golden is not None else load_golden(settings.golden_set_path)
    if limit is not None:
        cases = cases[:limit]

    if parser is None or retriever is None or generator is None:
        # Imported here so the eval package imports without the serving stack
        # configured (and so unit tests inject fakes without touching singletons).
        from hedonism_assistant.generation.generator import get_generator
        from hedonism_assistant.retrieval.query_parser import get_query_parser
        from hedonism_assistant.retrieval.retriever import get_retriever

        parser = parser or get_query_parser()
        retriever = retriever or get_retriever()
        generator = generator or get_generator()

    use_judge = settings.eval_judge_enabled if judge_enabled is None else judge_enabled
    active_judge = (judge or get_judge(settings)) if use_judge else None

    results = [
        await _evaluate_case(
            case,
            parser=parser,
            retriever=retriever,
            generator=generator,
            judge=active_judge,
            settings=settings,
        )
        for case in cases
    ]

    report = _aggregate(results, settings)
    Path(settings.eval_report_path).write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    logger.info(
        "eval_done",
        evaluated=report.evaluated,
        hit_at_k=report.hit_at_k,
        mrr=report.mrr,
        passed=report.passed,
    )
    return report


def _aggregate(results: list[CaseResult], settings: Settings) -> EvalReport:
    """Roll per-case results into dataset means, latency percentiles and the gate."""
    # Retrieval metrics are meaningful only for non-guardrail cases. Guardrail
    # cases (expected_fallback set) carry no relevance spec and are judged solely
    # on whether the pipeline took the expected branch (fallback_ok).
    scored = [r for r in results if r.expected_fallback is None]
    guardrail = [r for r in results if r.expected_fallback is not None]
    faith = [r.faithfulness for r in results if r.faithfulness is not None]
    relevancy = [r.answer_relevancy for r in results if r.answer_relevancy is not None]
    totals = [r.latency.total_ms for r in results]

    report = EvalReport(
        dataset_size=len(results),
        evaluated=len(scored),
        judged=sum(
            1 for r in results if r.faithfulness is not None or r.answer_relevancy is not None
        ),
        k=settings.rerank_top_k,
        hit_at_1=mean(r.hit_at_1 for r in scored),
        hit_at_3=mean(r.hit_at_3 for r in scored),
        hit_at_5=mean(r.hit_at_5 for r in scored),
        hit_at_k=mean(r.hit_at_k for r in scored),
        mrr=mean(r.reciprocal_rank for r in scored),
        faithfulness=mean(faith) if faith else None,
        answer_relevancy=mean(relevancy) if relevancy else None,
        avg_total_ms=mean(totals),
        p50_total_ms=percentile(totals, 50),
        p95_total_ms=percentile(totals, 95),
        avg_parse_ms=mean(r.latency.parse_ms for r in results),
        avg_retrieve_ms=mean(r.latency.retrieve_ms for r in results),
        avg_generate_ms=mean(r.latency.generate_ms for r in results),
        judge_calls=sum(
            (r.faithfulness is not None) + (r.answer_relevancy is not None) for r in results
        ),
        generation_calls=sum(1 for r in results if r.latency.generate_ms > 0),
        fallback_cases=len(guardrail),
        fallback_ok=sum(1 for r in guardrail if r.fallback_ok),
        thresholds=thresholds_of(settings),
        cases=results,
    )
    report.passed = evaluate_thresholds(report)
    return report


async def _collect(stream: AsyncIterator[str]) -> str:
    """Join an answer stream into a single string."""
    return "".join([delta async for delta in stream])


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Layer CLI overrides on top of the environment-backed settings."""
    overrides: dict[str, object] = {}
    if args.input is not None:
        overrides["golden_set_path"] = args.input
    if args.no_judge:
        overrides["eval_judge_enabled"] = False
    return get_settings().model_copy(update=overrides)


def _print_summary(report: EvalReport) -> None:
    print("\nEval summary")
    print("------------")
    print(f"  dataset size        : {report.dataset_size}")
    print(f"  retrieval cases     : {report.evaluated}")
    print(f"  judged cases        : {report.judged}")
    hits = f"{report.hit_at_1:.2f} / {report.hit_at_3:.2f} / {report.hit_at_5:.2f}"
    print(f"  hit@1 / @3 / @5     : {hits}")
    print(f"  hit@{report.k:<2}              : {report.hit_at_k:.2f}")
    print(f"  MRR                 : {report.mrr:.2f}")
    print(f"  faithfulness        : {_fmt(report.faithfulness)}")
    print(f"  answer relevancy    : {_fmt(report.answer_relevancy)}")
    print(f"  latency p50 / p95   : {report.p50_total_ms:.0f}ms / {report.p95_total_ms:.0f}ms")
    print(f"  fallbacks ok        : {report.fallback_ok}/{report.fallback_cases}")
    print(f"  PASSED              : {report.passed}")


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a (judge off)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-set eval harness (I-8).")
    parser.add_argument("--input", help="Golden-set JSONL path.")
    parser.add_argument("--limit", type=int, help="Cap the number of cases evaluated.")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge (retrieval metrics only, network-light).",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit non-zero when thresholds are missed (exploratory runs).",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="Human-readable logs instead of JSON.",
    )
    args = parser.parse_args()

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_output=not args.log_console)
    report = asyncio.run(run_eval(settings, limit=args.limit))
    _print_summary(report)
    if not report.passed and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
