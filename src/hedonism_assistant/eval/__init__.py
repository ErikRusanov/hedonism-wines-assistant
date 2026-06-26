"""Golden-set quality evaluation and regression harness (I-8).

Scores the live RAG pipeline over ``data/golden_set.jsonl``: retrieval metrics
(hit@k, MRR) plus LLM-judge answer quality (faithfulness, answer relevancy),
gated by configured thresholds. Run via ``python -m hedonism_assistant.eval.run``.

The orchestrator (:func:`hedonism_assistant.eval.run.run_eval`) is intentionally
*not* re-exported here: it is the ``-m``-executed module, and importing it from
the package ``__init__`` would trigger a double-import warning under ``runpy``.
Import it from :mod:`hedonism_assistant.eval.run` directly.
"""

from hedonism_assistant.eval.golden import GoldenCase, load_golden, matches
from hedonism_assistant.eval.judge import LLMJudge, get_judge
from hedonism_assistant.eval.report import CaseResult, EvalReport

__all__ = [
    "GoldenCase",
    "load_golden",
    "matches",
    "LLMJudge",
    "get_judge",
    "CaseResult",
    "EvalReport",
]
