"""Tests for the pure retrieval metrics (I-8)."""

from __future__ import annotations

from hedonism_assistant.eval.retrieval_metrics import hit_at_k, mean, reciprocal_rank


def test_hit_at_k_finds_relevant_within_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"c"}
    assert hit_at_k(ranked, relevant, 1) == 0.0
    assert hit_at_k(ranked, relevant, 3) == 1.0
    assert hit_at_k(ranked, relevant, 4) == 1.0


def test_hit_at_k_zero_when_none_relevant() -> None:
    assert hit_at_k(["a", "b"], {"z"}, 5) == 0.0
    assert hit_at_k(["a", "b"], set(), 5) == 0.0


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    # The earliest relevant wins even when several are relevant.
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 0.5


def test_reciprocal_rank_zero_when_absent() -> None:
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a", "b"], set()) == 0.0


def test_mean_handles_empty() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == 0.5
