"""Pure retrieval metrics over ranked id lists (I-8).

Computed on the final reranked order the user sees (``Retriever.retrieve``):

* **hit@k** — did any relevant wine land in the top ``k``? A blunt but honest
  "was the answer on the first screen" signal.
* **reciprocal rank** — ``1 / rank`` of the first relevant wine; its mean over
  the set is MRR, rewarding putting a good hit *high*, not just *somewhere*.

Both are plain arithmetic over (ranked ids, relevant ids) — no LLM, no Qdrant —
so they unit-test trivially and stay cheap to run on every case.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any of the first ``k`` ranked ids is relevant, else 0.0."""
    if not relevant_ids or k <= 0:
        return 0.0
    return 1.0 if any(rid in relevant_ids for rid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """``1 / rank`` (1-based) of the first relevant id; 0.0 if none is relevant."""
    if not relevant_ids:
        return 0.0
    for position, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / position
    return 0.0


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean, or 0.0 over an empty sequence (no cases to average)."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0
