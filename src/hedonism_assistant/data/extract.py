"""Extract canonical wine cards from captured product-page HTML (data track).

The catalogue is no longer scraped. Product-page HTML is captured by hand (see
``data/chrome_capture_prompt.md``) and saved as ``<slug>.html`` files in the
cache directory. This module walks those files, parses each into a permissive
:class:`RawWine` (:func:`parse_product`), turns it into a canonical :class:`Wine`
(:func:`normalize_wine`), drops non-wine or incomplete records, de-duplicates by
id, attaches the embedding passport, and streams the result to
``wines.enriched.jsonl`` -- the file the indexer (I-3) consumes. A report records
what was kept/dropped, the category mix and per-field coverage, so a run's
quality is visible at a glance.

Everything is deterministic and offline: no network, no LLM. Run it as a module::

    python -m hedonism_assistant.data.extract --log-console
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import ClassVar

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.data.models import RawWine
from hedonism_assistant.data.normalize import build_embedding_text, normalize_wine
from hedonism_assistant.data.parser import parse_product
from hedonism_assistant.logging_config import configure_logging, get_logger
from hedonism_assistant.models.wine import Wine

logger = get_logger(__name__)


@dataclass(slots=True)
class ExtractReport:
    """Counters, category mix and per-field coverage for one extraction run."""

    # Canonical-card fields whose fill-rate is worth tracking as a quality signal.
    COVERAGE_FIELDS: ClassVar[tuple[str, ...]] = (
        "producer",
        "country",
        "region",
        "sub_region",
        "vintage",
        "color",
        "grapes",
        "abv",
        "format_name",
        "critic_scores",
        "tasting_notes",
        "image_url",
        "is_vegan",
        "is_organic",
        "is_kosher",
        "is_alcohol_free",
        "embedding_text",
    )

    read: int = 0
    parse_failures: int = 0
    non_wine_skipped: int = 0
    dropped_incomplete: int = 0
    duplicates: int = 0
    written: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    field_coverage: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _is_populated(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            # For flag fields, "coverage" means the share of cards carrying it.
            return value
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


def _read_raw(settings: Settings, report: ExtractReport, limit: int | None) -> Iterator[RawWine]:
    """Yield :class:`RawWine` from each ``<slug>.html`` file in the input dir.

    The parser needs a page URL (for the slug); we synthesize one from the file
    name, which is the product slug by construction. Files that do not parse as a
    product page are counted and skipped rather than fatal.
    """
    input_dir = Path(settings.html_input_dir)
    for path in sorted(input_dir.glob("*.html")):
        if limit is not None and report.read >= limit:
            return
        report.read += 1
        slug = path.stem
        url = f"{settings.catalogue_base_url}/product/{slug}"
        try:
            raw = parse_product(path.read_text(encoding="utf-8", errors="ignore"), url)
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the run
            report.parse_failures += 1
            logger.warning("html_parse_failed", file=path.name, error=str(exc))
            continue
        if raw is None:
            report.parse_failures += 1
            logger.debug("not_a_product_page", file=path.name)
            continue
        yield raw


def _normalize_all(raws: Iterator[RawWine], report: ExtractReport) -> list[Wine]:
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


def run_extract(settings: Settings, *, limit: int | None = None) -> ExtractReport:
    """Execute a full extraction pass; writes the JSONL + report file."""
    report = ExtractReport()
    output_path = Path(settings.extract_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wines = _normalize_all(_read_raw(settings, report, limit), report)
    logger.info("normalize_done", kept=len(wines), read=report.read)

    with output_path.open("w", encoding="utf-8") as out:
        for wine in wines:
            wine.embedding_text = build_embedding_text(
                wine, notes_chars=settings.embedding_text_notes_chars
            )
            out.write(wine.model_dump_json() + "\n")
            report.written += 1

    report.finalize(wines)
    report_path = output_path.with_name("extract_report.json")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info(
        "extract_done",
        **{k: v for k, v in asdict(report).items() if k != "field_coverage"},
    )
    return report


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Layer CLI overrides on top of the environment-backed settings."""
    overrides: dict[str, object] = {}
    if args.input_dir is not None:
        overrides["html_input_dir"] = args.input_dir
    if args.output is not None:
        overrides["extract_output_path"] = args.output
    return get_settings().model_copy(update=overrides)


def _print_summary(report: ExtractReport) -> None:
    print("\nExtract summary")
    print("---------------")
    print(f"  read (html files) : {report.read}")
    print(f"  written (wines)   : {report.written}")
    print(f"  non-wine skipped  : {report.non_wine_skipped}")
    print(f"  dropped incomplete: {report.dropped_incomplete}")
    print(f"  duplicates        : {report.duplicates}")
    print(f"  parse failures    : {report.parse_failures}")
    print(f"  categories        : {report.category_counts}")
    print("  field coverage    :")
    for name, ratio in report.field_coverage.items():
        print(f"      {name:<16}: {ratio:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract canonical wine cards from captured product-page HTML."
    )
    parser.add_argument("--input-dir", help="Directory of captured <slug>.html files.")
    parser.add_argument("--output", help="Output enriched JSONL path.")
    parser.add_argument("--limit", type=int, help="Cap the number of HTML files processed.")
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="Human-readable logs instead of JSON.",
    )
    args = parser.parse_args()

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_output=not args.log_console)
    report = run_extract(settings, limit=args.limit)
    _print_summary(report)


if __name__ == "__main__":
    main()
