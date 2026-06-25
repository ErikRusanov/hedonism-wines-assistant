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

# A dense vector and a candidate paired with its vector. Vectors are L2-normalised
# by the embedder, so cosine similarity is just their dot product.
type Vector = list[float]
type ScoredCandidate = tuple[RetrievedWine, Vector]


def _cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _relevance(candidate: RetrievedWine) -> float:
    """Relevance signal for MMR: rerank score if present, else fusion score."""
    return candidate.rerank_score if candidate.rerank_score is not None else candidate.score


def _marginal_relevance(
    pair: ScoredCandidate, selected: list[ScoredCandidate], lambda_: float
) -> float:
    """MMR objective: ``λ·relevance − (1−λ)·max similarity to anything selected``."""
    candidate, vector = pair
    redundancy = max((_cosine(vector, chosen) for _, chosen in selected), default=0.0)
    return lambda_ * _relevance(candidate) - (1.0 - lambda_) * redundancy


def mmr_select(
    candidates: list[ScoredCandidate], *, lambda_: float, top_k: int
) -> list[RetrievedWine]:
    """Reorder ``(candidate, dense_vector)`` pairs by MMR; return up to ``top_k``."""
    pool = list(candidates)
    selected: list[ScoredCandidate] = []

    while pool and len(selected) < top_k:
        best = max(range(len(pool)), key=lambda i: _marginal_relevance(pool[i], selected, lambda_))
        selected.append(pool.pop(best))

    return [candidate for candidate, _ in selected]
