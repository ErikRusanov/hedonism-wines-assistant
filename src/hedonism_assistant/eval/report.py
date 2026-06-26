"""Eval report dataclasses, threshold gating and aggregation (I-8).

Mirrors the data-track reports (``IndexReport``): plain ``slots=True``
dataclasses serialised with ``dataclasses.asdict -> json.dumps(indent=2)`` to
``data/eval_report.json``. The aggregate ``EvalReport`` rolls up per-case results
into dataset means + latency percentiles and computes ``passed`` by comparing
each configured mean against its threshold (``None`` metrics — e.g. when judging
is off — are skipped, never failing the gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hedonism_assistant.config import Settings


@dataclass(slots=True)
class StageLatency:
    """Wall-clock per pipeline stage for one case, in milliseconds."""

    parse_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class CaseResult:
    """The full outcome of evaluating one golden case."""

    id: str
    question: str
    intent: str = ""
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    # Future seams (see eval/judge.py) — kept None for forward-compatible reports.
    context_precision: float | None = None
    context_recall: float | None = None
    latency: StageLatency = field(default_factory=StageLatency)
    expected_fallback: str | None = None  # set => guardrail case, excluded from retrieval means
    fallback: str | None = None  # guardrail branch actually taken, if any
    fallback_ok: bool | None = None  # matched the case's expected_fallback
    low_confidence: bool = False
    error: str | None = None


@dataclass(slots=True)
class EvalReport:
    """Dataset-level rollup persisted to ``data/eval_report.json``."""

    dataset_size: int = 0
    evaluated: int = 0  # retrieval cases scored (excludes guardrail/empty cases)
    judged: int = 0  # cases the LLM judge ran on
    k: int = 8
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_k: float = 0.0
    mrr: float = 0.0
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    avg_total_ms: float = 0.0
    p50_total_ms: float = 0.0
    p95_total_ms: float = 0.0
    avg_parse_ms: float = 0.0
    avg_retrieve_ms: float = 0.0
    avg_generate_ms: float = 0.0
    judge_calls: int = 0
    generation_calls: int = 0
    fallback_cases: int = 0
    fallback_ok: int = 0
    # Exact $-cost needs OpenRouter usage threaded through the client (out of
    # scope for the demo); left None as an explicit seam.
    est_cost_usd: float | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    cases: list[CaseResult] = field(default_factory=list)


def thresholds_of(settings: Settings) -> dict[str, float]:
    """The configured pass/fail bounds, keyed by the report metric name."""
    return {
        "hit_at_k": settings.eval_min_hit_at_k,
        "mrr": settings.eval_min_mrr,
        "faithfulness": settings.eval_min_faithfulness,
        "answer_relevancy": settings.eval_min_answer_relevancy,
    }


def evaluate_thresholds(report: EvalReport) -> bool:
    """Whether every metric with both a threshold and a value clears its bound."""
    for metric, bound in report.thresholds.items():
        value = getattr(report, metric, None)
        if value is not None and value < bound:
            return False
    return True


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (0.0 over an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return ordered[rank - 1]
