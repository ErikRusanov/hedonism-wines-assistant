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
