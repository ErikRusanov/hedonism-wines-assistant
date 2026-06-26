"""Tests for settings parsing, especially CSV list fields from the environment."""

import pytest

from hedonism_assistant.config import Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("a/b", ["a/b"]),
        ("a/b, c/d ,  e/f", ["a/b", "c/d", "e/f"]),
    ],
)
def test_fallback_models_parse_from_csv(
    raw: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENERATION_FALLBACK_MODELS", raw)
    settings = Settings(_env_file=None)
    assert settings.generation_fallback_models == expected


def test_embedding_defaults_are_local() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "BAAI/bge-base-en-v1.5"
    assert settings.embedding_dimensions == 768
    assert settings.embedding_device == ""
    assert settings.embedding_batch_size == 64


def test_indexing_defaults_match_frozen_contract() -> None:
    settings = Settings(_env_file=None)
    assert settings.qdrant_dense_vector_name == "dense"
    assert settings.qdrant_sparse_vector_name == "sparse"
    assert settings.sparse_enabled is True
    assert settings.sparse_encoder_path == "data/sparse_encoder.json"
    assert settings.index_batch_size == 128


def test_cors_allow_origins_default_is_permissive() -> None:
    assert Settings(_env_file=None).cors_allow_origins == ["*"]


def test_cors_allow_origins_parse_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
    settings = Settings(_env_file=None)
    assert settings.cors_allow_origins == ["https://a.example", "https://b.example"]
