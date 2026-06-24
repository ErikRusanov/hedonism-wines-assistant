"""Scrape orchestrator and CLI for the data track (I-1).

Pipeline: discover product URLs (either from a file or by paginating the
``/wines`` listing), fetch each page through a real browser (Playwright Chromium
-- the only thing that gets past Cloudflare), parse it into a :class:`RawWine`,
and merge the results into ``wines.raw.jsonl``. A coverage report -- how many
products were found/fetched/kept and how completely each field is populated -- is
written alongside the data and logged, so the quality of a run is visible at a
glance.

The output file is the single, cumulative dataset. By default a run is
**incremental**: products already present are skipped (no re-fetch, no
duplicates) and only new ones are appended, so you can grow the catalogue across
many runs without rebuilding it each time. ``--rewrite`` forces a clean rebuild
from scratch. Either way the JSONL is written atomically (temp file +
``os.replace``) and never clobbered by an empty/blocked run, so a failed crawl
leaves the previous dataset intact.

Run it as a module::

    python -m hedonism_assistant.data.scrape --log-console           # whole catalogue
    python -m hedonism_assistant.data.scrape --urls-file urls.txt    # just these URLs
    python -m hedonism_assistant.data.scrape --rewrite --log-console # clean rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.data.browser import BrowserFetcher
from hedonism_assistant.data.listing import discover_via_listing
from hedonism_assistant.data.models import RawWine
from hedonism_assistant.data.parser import parse_product
from hedonism_assistant.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass
class ScrapeReport:
    """Counters and per-field coverage describing one scrape run."""

    # Fields whose fill-rate is worth tracking as a data-quality signal.
    COVERAGE_FIELDS: ClassVar[tuple[str, ...]] = (
        "name",
        "sku",
        "producer",
        "region",
        "country",
        "vintage",
        "color",
        "grapes",
        "abv",
        "bottle_size_ml",
        "price",
        "availability",
        "tasting_notes",
        "image_url",
        "critic_scores",
    )

    discovered: int = 0
    skipped_existing: int = 0
    fetched: int = 0
    from_cache: int = 0
    parse_failures: int = 0
    fetch_errors: int = 0
    non_wine_skipped: int = 0
    written: int = 0  # new wines added this run
    total: int = 0  # total records in the output file after merge
    field_coverage: dict[str, float] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of fetched pages served from the on-disk cache."""
        return round(self.from_cache / self.fetched, 3) if self.fetched else 0.0

    @staticmethod
    def _is_populated(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, list | str | dict):
            return len(value) > 0
        return True

    @classmethod
    def _coverage(cls, records: Sequence[RawWine]) -> dict[str, float]:
        if not records:
            return dict.fromkeys(cls.COVERAGE_FIELDS, 0.0)
        return {
            name: round(sum(cls._is_populated(getattr(r, name)) for r in records) / len(records), 3)
            for name in cls.COVERAGE_FIELDS
        }

    def finalize(self, records: Sequence[RawWine]) -> None:
        """Populate ``field_coverage`` from the merged records."""
        self.total = len(records)
        self.field_coverage = self._coverage(records)


def _read_existing(output_path: Path) -> dict[str, RawWine]:
    """Load already-scraped records keyed by URL; empty if the file is absent.

    Malformed lines are skipped rather than fatal, so a partially-written file
    from an interrupted run never blocks the next one.
    """
    if not output_path.exists():
        return {}
    by_url: dict[str, RawWine] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            wine = RawWine.model_validate_json(line)
        except ValidationError:
            continue
        by_url[wine.url] = wine
    return by_url


def _safe_write_jsonl(output_path: Path, records: Sequence[RawWine]) -> bool:
    """Atomically write ``records`` to ``output_path``; never clobber on empty.

    A blocked or empty run must not destroy a previously-good dataset, so when
    there is nothing to write we leave any existing file untouched. Otherwise we
    write to a temp file and ``os.replace`` it into place (atomic on POSIX).
    """
    if not records:
        logger.warning(
            "output_unchanged",
            reason="no records produced (blocked or empty run); previous output kept",
            output=str(output_path),
        )
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(record.model_dump_json() + "\n")
    os.replace(tmp_path, output_path)
    return True


def _write_report(settings: Settings, report: ScrapeReport) -> None:
    report_path = Path(settings.scrape_output_path).with_name("scrape_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _keep(report: ScrapeReport, wine: RawWine | None, *, wines_only: bool) -> RawWine | None:
    """Apply the parse/wines-only filters, updating ``report`` counters."""
    if wine is None:
        report.parse_failures += 1
        return None
    if wines_only and not wine.is_wine:
        report.non_wine_skipped += 1
        return None
    report.written += 1
    return wine


async def _handle_one(
    fetcher: BrowserFetcher, url: str
) -> tuple[str, RawWine | None, bool, Exception | None]:
    try:
        result = await fetcher.get_product_html(url)
        wine = parse_product(result.html, url)
        return url, wine, result.from_cache, None
    except Exception as exc:  # noqa: BLE001 - one bad page must not sink the run
        return url, None, False, exc


async def run_scrape(
    settings: Settings,
    *,
    seed_urls: Sequence[str] | None = None,
    rewrite: bool = False,
) -> ScrapeReport:
    """Execute a scrape and return its report; merges into the output JSONL.

    Discovery source:

    * ``seed_urls`` -- fetch exactly these (skips discovery); read from a file.
    * default -- paginate ``/wines?pg=N`` for the **whole** catalogue (~7.9k).

    Output is incremental: products already in the JSONL are skipped and only new
    ones are appended. ``rewrite=True`` ignores existing output and rebuilds it.
    """
    report = ScrapeReport()
    output_path = Path(settings.scrape_output_path)
    existing = {} if rewrite else _read_existing(output_path)

    async with BrowserFetcher(settings) as fetcher:
        if seed_urls is not None:
            urls = list(seed_urls)
            if settings.scrape_max_products is not None:
                urls = urls[: settings.scrape_max_products]
        else:
            urls = await discover_via_listing(
                fetcher.fetch_text,
                settings.scrape_base_url,
                max_products=settings.scrape_max_products,
            )
        report.discovered = len(urls)
        if not urls:
            # Almost always Cloudflare blocking the listing. Do NOT write output
            # -- that would wipe a previously-good dataset.
            logger.warning(
                "no_products_discovered",
                hint="listing blocked or empty; existing output left untouched",
            )
            _write_report(settings, report)
            return report

        todo = [u for u in urls if u not in existing]
        report.skipped_existing = len(urls) - len(todo)
        logger.info(
            "scrape_start",
            discovered=report.discovered,
            to_fetch=len(todo),
            skipped_existing=report.skipped_existing,
            output=str(output_path),
        )

        kept: list[RawWine] = []
        tasks = [asyncio.create_task(_handle_one(fetcher, url)) for url in todo]
        for index, future in enumerate(asyncio.as_completed(tasks), start=1):
            url, wine, from_cache, error = await future
            if error is not None:
                report.fetch_errors += 1
                logger.warning("fetch_error", url=url, error=str(error))
                continue
            report.fetched += 1
            report.from_cache += int(from_cache)
            if (wanted := _keep(report, wine, wines_only=settings.scrape_wines_only)) is not None:
                kept.append(wanted)
            if index % 250 == 0:
                logger.info("scrape_progress", processed=index, total=len(todo))

    merged = list(existing.values()) + kept
    if kept or rewrite:
        _safe_write_jsonl(output_path, merged)
    else:
        logger.info("output_unchanged", reason="no new products this run", output=str(output_path))
    report.finalize(merged)
    _write_report(settings, report)
    logger.info("scrape_done", **{k: v for k, v in asdict(report).items() if k != "field_coverage"})
    return report


def _read_urls_file(path: str) -> list[str]:
    """Read product URLs from a file: one per line, blanks and #comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    stripped = (line.strip() for line in lines)
    return [s for s in stripped if s and not s.startswith("#")]


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Layer CLI overrides on top of the environment-backed settings."""
    overrides: dict[str, object] = {}
    if args.limit is not None:
        overrides["scrape_max_products"] = args.limit
    if args.output is not None:
        overrides["scrape_output_path"] = args.output
    if args.delay is not None:
        overrides["scrape_request_delay_seconds"] = args.delay
    if args.concurrency is not None:
        overrides["scrape_max_concurrency"] = args.concurrency
    if args.headful:
        overrides["scrape_browser_headless"] = False
    if args.browser_profile is not None:
        overrides["scrape_browser_user_data_dir"] = args.browser_profile
    return get_settings().model_copy(update=overrides)


def _print_summary(report: ScrapeReport) -> None:
    print("\nScrape summary")
    print("--------------")
    print(f"  discovered      : {report.discovered}")
    print(f"  skipped (had)   : {report.skipped_existing}")
    print(
        f"  fetched         : {report.fetched} "
        f"({report.from_cache} from cache, {report.cache_hit_rate:.0%} hit rate)"
    )
    print(f"  new wines       : {report.written}")
    print(f"  total in file   : {report.total}")
    print(f"  non-wine skipped: {report.non_wine_skipped}")
    print(f"  parse failures  : {report.parse_failures}")
    print(f"  fetch errors    : {report.fetch_errors}")
    print("  field coverage  :")
    for name, ratio in report.field_coverage.items():
        print(f"      {name:<16}: {ratio:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the Hedonism wines catalogue (I-1).")
    parser.add_argument("--limit", type=int, help="Cap the number of products processed.")
    parser.add_argument("--delay", type=float, help="Seconds to wait between requests.")
    parser.add_argument("--concurrency", type=int, help="Max concurrent requests.")
    parser.add_argument("--output", help="Output JSONL path (the cumulative dataset).")
    parser.add_argument(
        "--urls-file",
        help="Fetch exactly the product URLs in this file (one per line); skips listing discovery.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rebuild the output from scratch instead of merging into the existing file.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Show the browser window (lets you clear an interactive Cloudflare "
        "challenge by hand).",
    )
    parser.add_argument(
        "--browser-profile",
        help="A Chrome user-data dir to persist the cf_clearance cookie across runs.",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="Human-readable logs instead of JSON.",
    )
    args = parser.parse_args()

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_output=not args.log_console)
    seed_urls = _read_urls_file(args.urls_file) if args.urls_file else None
    report = asyncio.run(run_scrape(settings, seed_urls=seed_urls, rewrite=args.rewrite))
    _print_summary(report)


if __name__ == "__main__":
    main()
