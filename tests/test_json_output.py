"""Tests for the tolerant LLM-JSON loader."""

from __future__ import annotations

import json

import pytest

from hedonism_assistant.llm.json_output import loads_json


def test_plain_json_object() -> None:
    assert loads_json('{"ok": true}') == {"ok": True}


def test_strips_json_code_fence() -> None:
    # The exact shape Anthropic models return through OpenRouter.
    assert loads_json('```json\n{\n  "ok": true\n}\n```') == {"ok": True}


def test_strips_bare_code_fence() -> None:
    assert loads_json('```\n{"a": 1}\n```') == {"a": 1}


def test_recovers_object_wrapped_in_prose() -> None:
    raw = 'Here is the result: {"intent": "recommendation"} — hope that helps!'
    assert loads_json(raw) == {"intent": "recommendation"}


def test_parses_non_object_json() -> None:
    assert loads_json("[1, 2, 3]") == [1, 2, 3]


def test_empty_string_raises_json_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        loads_json("")


def test_unparseable_raises_json_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        loads_json("not json at all")
