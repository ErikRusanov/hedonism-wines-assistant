"""Tests for the enrich orchestrator (I-2): normalize, dedupe, report, output."""

import json
from pathlib import Path

from hedonism_assistant.config import Settings
from hedonism_assistant.data.enrich import run_enrich
from hedonism_assistant.data.models import RawWine


def _raw_line(**overrides: object) -> str:
    base: dict[str, object] = {
        "url": "https://hedonism.co.uk/product/a",
        "slug": "a",
        "sku": "HEDA",
        "name": "A Wine",
        "section": "Wines",
        "bottle_size_ml": 750,
        "price": 20.0,
    }
    base.update(overrides)
    return RawWine(**base).model_dump_json()


def _settings(inp: Path, outp: Path) -> Settings:
    return Settings(
        enrich_input_path=str(inp),
        enrich_output_path=str(outp),
        enrich_use_llm=False,
    )


async def test_run_enrich_counts_and_output(tmp_path: Path) -> None:
    inp = tmp_path / "raw.jsonl"
    outp = tmp_path / "enriched.jsonl"
    lines = [
        _raw_line(),  # A (still)
        _raw_line(),  # duplicate of A (same sku)
        _raw_line(
            url="https://hedonism.co.uk/product/b",
            slug="b",
            sku="HEDB",
            name="Bollinger NV",
            region="Champagne",
        ),  # sparkling
        _raw_line(
            url="https://hedonism.co.uk/product/s",
            slug="s",
            sku="HEDS",
            name="Some Whisky",
            section="Spirits",
        ),  # non-wine
        _raw_line(
            url="https://hedonism.co.uk/product/c",
            slug="c",
            sku="HEDC",
            name="Incomplete",
            price=None,
        ),  # incomplete (no price)
        "{ this is not valid json",  # parse failure
    ]
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = await run_enrich(_settings(inp, outp))

    assert report.read == 6
    assert report.written == 2
    assert report.duplicates == 1
    assert report.non_wine_skipped == 1
    assert report.dropped_incomplete == 1
    assert report.parse_failures == 1
    assert report.llm_enriched == 0
    assert report.category_counts == {"still": 1, "sparkling": 1}

    written = [json.loads(line) for line in outp.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    assert all(card["embedding_text"] for card in written)
    assert (outp.parent / "enrich_report.json").exists()


async def test_run_enrich_respects_limit(tmp_path: Path) -> None:
    inp = tmp_path / "raw.jsonl"
    outp = tmp_path / "enriched.jsonl"
    inp.write_text(
        "\n".join(_raw_line(sku=f"HED{i}", slug=str(i)) for i in range(5)) + "\n",
        encoding="utf-8",
    )

    report = await run_enrich(_settings(inp, outp), limit=3)

    assert report.read == 3
    assert report.written == 3
