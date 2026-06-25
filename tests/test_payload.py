"""Tests for the Qdrant payload builder and critic-score normalisation (I-3)."""

from hedonism_assistant.models.wine import CriticScore
from hedonism_assistant.vector_store.payload import (
    MAX_CRITIC_SCORE_FIELD,
    build_payload,
    normalize_critic_score,
)
from tests.fixtures.wines import make_wine


def test_normalize_critic_score_rescales_to_100() -> None:
    assert normalize_critic_score(96, 100) == 96.0
    assert normalize_critic_score(18, 20) == 90.0


def test_build_payload_reconstructs_card_fields() -> None:
    wine = make_wine(country="France", region="Bordeaux", vintage=2015, in_bond=True)
    payload = build_payload(wine)
    assert payload["id"] == wine.id
    assert payload["country"] == "France"
    assert payload["region"] == "Bordeaux"
    assert payload["vintage"] == 2015
    assert payload["in_bond"] is True
    assert payload["category"] == "still"


def test_max_critic_score_takes_max_across_mixed_scales() -> None:
    wine = make_wine(
        critic_scores=[
            CriticScore(critic="Vinous", score=95, scale=100),  # -> 95
            CriticScore(critic="Jancis Robinson", score=18, scale=20),  # -> 90
        ]
    )
    payload = build_payload(wine)
    assert payload[MAX_CRITIC_SCORE_FIELD] == 95.0


def test_20_point_score_outranks_lower_100_point_score() -> None:
    wine = make_wine(
        critic_scores=[
            CriticScore(critic="Decanter", score=88, scale=100),  # -> 88
            CriticScore(critic="Jancis Robinson", score=19, scale=20),  # -> 95
        ]
    )
    assert build_payload(wine)[MAX_CRITIC_SCORE_FIELD] == 95.0


def test_field_omitted_when_no_critic_scores() -> None:
    payload = build_payload(make_wine(critic_scores=[]))
    assert MAX_CRITIC_SCORE_FIELD not in payload
