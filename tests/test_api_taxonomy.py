"""Tests for startup taxonomy loading from the index (I-7).

A non-empty index yields a populated taxonomy; an unreachable/empty index
degrades to an empty pass-through taxonomy so the app always starts.
"""

from __future__ import annotations

from hedonism_assistant.api.taxonomy import load_taxonomy
from tests.fixtures.wines import sample_wines


class _ScrollStore:
    """Fake store whose ``scroll_payloads`` yields serialised wine payloads."""

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads

    async def scroll_payloads(self, *, batch: int = 256):
        for payload in self._payloads:
            yield payload


class _BrokenStore:
    async def scroll_payloads(self, *, batch: int = 256):
        raise RuntimeError("qdrant unreachable")
        yield  # pragma: no cover - makes this an async generator


async def test_load_taxonomy_builds_from_payloads() -> None:
    payloads = [w.model_dump(mode="json") for w in sample_wines()]
    taxonomy = await load_taxonomy(_ScrollStore(payloads))

    assert taxonomy.countries  # non-empty
    assert "Bordeaux" in taxonomy.regions


async def test_load_taxonomy_empty_index_is_pass_through() -> None:
    taxonomy = await load_taxonomy(_ScrollStore([]))
    assert taxonomy.countries == frozenset()
    assert taxonomy.regions == frozenset()


async def test_load_taxonomy_degrades_on_failure() -> None:
    taxonomy = await load_taxonomy(_BrokenStore())
    assert taxonomy.countries == frozenset()
    assert taxonomy.grapes == frozenset()
