"""Normalize & enrich orchestrator and CLI for the data track (I-2).

Pipeline: read the permissive scrape output (``wines.raw.jsonl``), turn each
record into a canonical :class:`Wine` (:func:`normalize_wine`), drop incomplete
or non-wine records, de-duplicate by id, optionally enrich with the LLM, attach
the embedding passport, and stream the result to ``wines.enriched.jsonl`` -- the
file I-3 indexes into Qdrant. A report mirroring the scrape report records what
was kept/dropped, the category mix and per-field coverage, so a run's quality is
visible at a glance.

Run it as a module::

    python -m hedonism_assistant.data.enrich --log-console
    python -m hedonism_assistant.data.enrich --use-llm   # add style/pairing tags

Deterministic by default (no network); ``--use-llm`` turns on the optional tag
enrichment via the utility model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import ClassVar

from pydantic import ValidationError

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.data.enricher import LlmEnricher
from hedonism_assistant.data.models import RawWine
from hedonism_assistant.data.normalize import build_embedding_text, normalize_wine
from hedonism_assistant.llm.openrouter import get_openrouter_client
from hedonism_assistant.logging_config import configure_logging, get_logger
from hedonism_assistant.models.wine import Wine

logger = get_logger(__name__)


@dataclass
class EnrichReport:
    """Counters, category mix and per-field coverage for one enrichment run."""

    # Canonical-card fields whose fill-rate is worth tracking as a quality signal.
    COVERAGE_FIELDS: ClassVar[tuple[str, ...]] = (
        "producer",
        "country",
        "region",
        "vintage",
        "color",
        "grapes",
        "abv",
        "format_name",
        "critic_scores",
        "tasting_notes",
        "image_url",
        "style_tags",
        "food_pairings",
        "embedding_text",
    )

    read: int = 0
    parse_failures: int = 0
    non_wine_skipped: int = 0
    dropped_incomplete: int = 0
    duplicates: int = 0
    llm_enriched: int = 0
    written: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    field_coverage: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _is_populated(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, list | str | dict):
            return len(value) > 0
        return True

    def finalize(self, wines: Sequence[Wine]) -> None:
        """Populate the category mix and per-field coverage from kept cards."""
        self.category_counts = dict(Counter(w.category.value for w in wines))
        if not wines:
            self.field_coverage = dict.fromkeys(self.COVERAGE_FIELDS, 0.0)
            return
        self.field_coverage = {
            name: round(fmean(self._is_populated(getattr(w, name)) for w in wines), 3)
            for name in self.COVERAGE_FIELDS
        }


def _read_raw(path: Path, report: EnrichReport, limit: int | None) -> Iterator[RawWine]:
    """Yield :class:`RawWine` records from JSONL, counting malformed lines."""
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if limit is not None and report.read >= limit:
                return
            report.read += 1
            try:
                yield RawWine.model_validate_json(stripped)
            except ValidationError as exc:
                report.parse_failures += 1
                logger.warning("raw_parse_failed", error=str(exc))


def _enrichment_fingerprint(wine: Wine) -> tuple[object, ...]:
    """The fields the LLM pass may touch, for detecting whether it changed a card."""
    return (wine.color, tuple(wine.style_tags), tuple(wine.food_pairings))


def _normalize_all(raws: Iterator[RawWine], report: EnrichReport) -> list[Wine]:
    """Normalize, drop non-wine/incomplete records, and de-duplicate by id."""
    seen: set[str] = set()
    wines: list[Wine] = []
    for raw in raws:
        if not raw.is_wine:
            report.non_wine_skipped += 1
            continue
        wine = normalize_wine(raw)
        if wine is None:
            report.dropped_incomplete += 1
            continue
        if wine.id in seen:
            report.duplicates += 1
            continue
        seen.add(wine.id)
        wines.append(wine)
    return wines


async def run_enrich(settings: Settings, *, limit: int | None = None) -> EnrichReport:
    """Execute a full normalize/enrich pass; writes the JSONL + report file."""
    report = EnrichReport()
    input_path = Path(settings.enrich_input_path)
    output_path = Path(settings.enrich_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wines = _normalize_all(_read_raw(input_path, report, limit), report)
    logger.info("normalize_done", kept=len(wines), read=report.read)

    if settings.enrich_use_llm and wines:
        enricher = LlmEnricher(get_openrouter_client(), settings)
        before = [_enrichment_fingerprint(w) for w in wines]
        wines = await enricher.enrich_many(wines)
        report.llm_enriched = sum(
            _enrichment_fingerprint(w) != snap for w, snap in zip(wines, before, strict=True)
        )
        logger.info("llm_enrich_done", enriched=report.llm_enriched)

    with output_path.open("w", encoding="utf-8") as out:
        for wine in wines:
            wine.embedding_text = build_embedding_text(
                wine, notes_chars=settings.embedding_text_notes_chars
            )
            out.write(wine.model_dump_json() + "\n")
            report.written += 1

    report.finalize(wines)
    report_path = output_path.with_name("enrich_report.json")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info(
        "enrich_done",
        **{k: v for k, v in asdict(report).items() if k != "field_coverage"},
    )
    return report


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Layer CLI overrides on top of the environment-backed settings."""
    overrides: dict[str, object] = {}
    if args.input is not None:
        overrides["enrich_input_path"] = args.input
    if args.output is not None:
        overrides["enrich_output_path"] = args.output
    if args.use_llm:
        overrides["enrich_use_llm"] = True
    return get_settings().model_copy(update=overrides)


def _print_summary(report: EnrichReport) -> None:
    print("\nEnrich summary")
    print("--------------")
    print(f"  read              : {report.read}")
    print(f"  written (wines)   : {report.written}")
    print(f"  non-wine skipped  : {report.non_wine_skipped}")
    print(f"  dropped incomplete: {report.dropped_incomplete}")
    print(f"  duplicates        : {report.duplicates}")
    print(f"  parse failures    : {report.parse_failures}")
    print(f"  llm enriched      : {report.llm_enriched}")
    print(f"  categories        : {report.category_counts}")
    print("  field coverage    :")
    for name, ratio in report.field_coverage.items():
        print(f"      {name:<16}: {ratio:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize & enrich scraped wines into canonical cards (I-2)."
    )
    parser.add_argument("--input", help="Input raw JSONL path.")
    parser.add_argument("--output", help="Output enriched JSONL path.")
    parser.add_argument("--limit", type=int, help="Cap the number of records processed.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enrich style/food-pairing tags with the utility model (needs an API key).",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="Human-readable logs instead of JSON.",
    )
    args = parser.parse_args()

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_output=not args.log_console)
    report = asyncio.run(run_enrich(settings, limit=args.limit))
    _print_summary(report)


if __name__ == "__main__":
    main()
