"""Maximal Marginal Relevance diversification of the final result list (I-5).

MMR trades relevance against diversity, greedily picking the candidate that
maximises ``λ·relevance − (1−λ)·max similarity to anything already picked``. It
is an optional, config-gated stage (``mmr_enabled``, default off): it competes
with the reranker for ordering, so it runs *after* reranking, only when asked.

Candidate dense vectors are L2-normalised by the embedder, so cosine similarity
is just their dot product.
"""

from __future__ import annotations

from hedonism_assistant.models.wine import RetrievedWine


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _relevance(candidate: RetrievedWine) -> float:
    """Relevance signal for MMR: rerank score if present, else fusion score."""
    return candidate.rerank_score if candidate.rerank_score is not None else candidate.score


def mmr_select(
    candidates: list[tuple[RetrievedWine, list[float]]],
    *,
    lambda_: float,
    top_k: int,
) -> list[RetrievedWine]:
    """Reorder ``(candidate, dense_vector)`` pairs by MMR; return up to ``top_k``."""
    pool = list(candidates)
    selected: list[tuple[RetrievedWine, list[float]]] = []

    while pool and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")
        for i, (candidate, vector) in enumerate(pool):
            diversity = max((_dot(vector, v) for _, v in selected), default=0.0)
            score = lambda_ * _relevance(candidate) - (1.0 - lambda_) * diversity
            if score > best_score:
                best_score = score
                best_index = i
        selected.append(pool.pop(best_index))

    return [candidate for candidate, _ in selected]
