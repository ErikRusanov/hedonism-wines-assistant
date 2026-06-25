"""Qdrant payload builder and critic-score normalisation (frozen I-3/I-5 contract).

The payload stores the full serialised :class:`Wine` (so retrieval can rebuild
the card without a second lookup) plus one derived, filterable field:
``max_critic_score_100``. The catalogue mixes critic scales (Parker/Vinous on
100, Jancis Robinson on 20), so a single numeric index over raw scores would be
meaningless — we normalise every score to a 100-point scale first and index the
maximum.
"""

from __future__ import annotations

from typing import Any, Final

from hedonism_assistant.models.wine import Wine

# Derived payload key holding the best critic score on a unified 100-pt scale.
MAX_CRITIC_SCORE_FIELD: Final = "max_critic_score_100"


def normalize_critic_score(score: float, scale: int) -> float:
    """Rescale a critic score to a 100-point scale (frozen I-5 source of truth)."""
    return score * 100 / scale


def build_payload(wine: Wine) -> dict[str, Any]:
    """Serialise a wine for Qdrant, adding the unified ``max_critic_score_100``.

    The field is omitted entirely when the wine has no critic scores, so the
    numeric payload index never sees a sentinel value.
    """
    payload = wine.model_dump(mode="json")
    if wine.critic_scores:
        payload[MAX_CRITIC_SCORE_FIELD] = max(
            normalize_critic_score(s.score, s.scale) for s in wine.critic_scores
        )
    return payload
