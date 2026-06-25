"""Tests for the deterministic BM25-style SparseEncoder (I-3, shared with I-5)."""

from pathlib import Path

from hedonism_assistant.vector_store.sparse import SparseEncoder


def test_encode_is_deterministic_across_instances() -> None:
    # Two independently-fitted encoders over the same corpus must agree exactly,
    # and the token->dimension hash must be stable (not Python's salted hash()).
    corpus = ["red bordeaux wine", "white burgundy wine"]
    a = SparseEncoder.fit(corpus)
    b = SparseEncoder.fit(corpus)
    assert a.encode("red bordeaux wine") == b.encode("red bordeaux wine")


def test_tokenization_drops_single_chars_and_splits_non_alnum() -> None:
    encoder = SparseEncoder()
    indices, values = encoder.encode("A full-bodied, 2015 wine!")
    # "a" is dropped (length 1); "full", "bodied", "2015", "wine" remain.
    assert len(indices) == 4
    assert len(values) == 4


def test_duplicate_tokens_aggregate_into_one_dimension() -> None:
    encoder = SparseEncoder()
    single = encoder.encode("wine")
    repeated = encoder.encode("wine wine wine")
    # Same single dimension; the repeated text carries more saturated weight.
    assert len(repeated[0]) == 1
    assert single[0] == repeated[0]
    assert repeated[1][0] > single[1][0]


def test_fallback_without_fit_uses_saturated_tf() -> None:
    encoder = SparseEncoder()  # empty IDF
    indices, values = encoder.encode("merlot")
    assert len(indices) == 1
    # tf=1 -> 1 * (k1+1)/(1+k1) = 2.5/2.5 = 1.0
    assert values[0] == 1.0


def test_idf_weights_rare_terms_above_common_ones() -> None:
    corpus = ["common term", "common term", "common rare"]
    encoder = SparseEncoder.fit(corpus)
    common_idx, common_val = encoder.encode("common")
    rare_idx, rare_val = encoder.encode("rare")
    assert rare_val[0] > common_val[0]


def test_unseen_token_after_fit_is_dropped() -> None:
    encoder = SparseEncoder.fit(["only these words"])
    indices, values = encoder.encode("unseen")
    assert indices == []
    assert values == []


def test_save_load_round_trip(tmp_path: Path) -> None:
    encoder = SparseEncoder.fit(["red bordeaux", "white burgundy"])
    path = tmp_path / "sparse.json"
    encoder.save(path)
    loaded = SparseEncoder.load(path)
    assert loaded == encoder
    assert loaded.encode("red bordeaux") == encoder.encode("red bordeaux")
