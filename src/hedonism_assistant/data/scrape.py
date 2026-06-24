"""Scrape orchestrator and CLI for the data track (I-1).

Pipeline: discover product URLs from the sitemap, fetch each page politely (with
an on-disk cache), parse it into a :class:`RawWine`, keep the wines, and write
the results to ``wines.raw.jsonl``. A coverage report -- how many products were
found/fetched/kept and how completely each field is populated -- is written
alongside the data and logged, so the quality of a run is visible at a glance.

Two safety properties matter when the live site is behind anti-bot protection
that can start refusing us mid-project:

* **A failed run never destroys good data.** The JSONL is written atomically
  (temp file + ``os.replace``) and only when at least one record was produced, so
  a blocked sitemap (0 discovered) leaves the previous dataset untouched.
* **An offline rebuild path.** ``--from-cache`` re-parses whatever HTML is already
  cached without touching the network, so the normalization stage can keep
  iterating while the site is blocking fresh fetches.

Run it as a module::

    python -m hedonism_assistant.data.scrape --limit 50 --log-console
    python -m hedonism_assistant.data.scrape --from-cache --log-console

Re-runs are idempotent: cached pages are reused, so only unseen products hit the
network.
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

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.data.fetcher import Fetcher
from hedonism_assistant.data.models import RawWine
from hedonism_assistant.data.parser import parse_product, product_markup_missing
from hedonism_assistant.data.sitemap import discover_product_urls
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
    fetched: int = 0
    from_cache: int = 0
    parse_failures: int = 0
    fetch_errors: int = 0
    non_wine_skipped: int = 0
    written: int = 0
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
        """Populate ``field_coverage`` from the kept records."""
        self.field_coverage = self._coverage(records)


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
    fetcher: Fetcher, url: str
) -> tuple[str, RawWine | None, bool, Exception | None]:
    try:
        result = await fetcher.get_product_html(url, needs_render=product_markup_missing)
        wine = parse_product(result.html, url)
        return url, wine, result.from_cache, None
    except Exception as exc:  # noqa: BLE001 - one bad page must not sink the run
        return url, None, False, exc


async def run_scrape(settings: Settings) -> ScrapeReport:
    """Execute a full scrape and return its report; writes JSONL + report file."""
    report = ScrapeReport()
    output_path = Path(settings.scrape_output_path)

    async with Fetcher(settings) as fetcher:
        urls = await discover_product_urls(
            fetcher.fetch_text,
            settings.scrape_sitemap_url,
            max_products=settings.scrape_max_products,
        )
        report.discovered = len(urls)
        if not urls:
            # Almost always the sitemap being blocked (403/404). Do NOT write
            # output -- that would wipe a previously-good dataset.
            logger.warning(
                "no_products_discovered",
                hint="sitemap blocked or empty; existing output left untouched",
            )
            _write_report(settings, report)
            return report

        logger.info("scrape_start", discovered=report.discovered, output=str(output_path))
        kept: list[RawWine] = []
        tasks = [asyncio.create_task(_handle_one(fetcher, url)) for url in urls]
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
                logger.info("scrape_progress", processed=index, total=report.discovered)

    _safe_write_jsonl(output_path, kept)
    report.finalize(kept)
    _write_report(settings, report)
    logger.info("scrape_done", **{k: v for k, v in asdict(report).items() if k != "field_coverage"})
    return report


def run_from_cache(settings: Settings) -> ScrapeReport:
    """Rebuild the JSONL from already-cached HTML, without touching the network.

    Lets the normalization stage (I-2) iterate on whatever has been scraped so far
    even while the live site is blocking fresh fetches.
    """
    report = ScrapeReport()
    output_path = Path(settings.scrape_output_path)
    cache_dir = Path(settings.scrape_cache_dir) / "html"
    pages = sorted(cache_dir.glob("*.html"))
    if settings.scrape_max_products is not None:
        pages = pages[: settings.scrape_max_products]
    report.discovered = len(pages)
    logger.info("cache_rebuild_start", pages=report.discovered, cache=str(cache_dir))

    kept: list[RawWine] = []
    for page in pages:
        report.fetched += 1
        report.from_cache += 1
        url = f"{settings.scrape_base_url}/product/{page.stem}"
        wine = parse_product(page.read_text(encoding="utf-8"), url)
        if (wanted := _keep(report, wine, wines_only=settings.scrape_wines_only)) is not None:
            kept.append(wanted)

    _safe_write_jsonl(output_path, kept)
    report.finalize(kept)
    _write_report(settings, report)
    logger.info(
        "cache_rebuild_done",
        **{k: v for k, v in asdict(report).items() if k != "field_coverage"},
    )
    return report


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Layer CLI overrides on top of the environment-backed settings."""
    overrides: dict[str, object] = {}
    if args.limit is not None:
        overrides["scrape_max_products"] = args.limit
    if args.output is not None:
        overrides["scrape_output_path"] = args.output
    if args.sitemap is not None:
        overrides["scrape_sitemap_url"] = args.sitemap
    if args.delay is not None:
        overrides["scrape_request_delay_seconds"] = args.delay
    if args.concurrency is not None:
        overrides["scrape_max_concurrency"] = args.concurrency
    if args.impersonate is not None:
        overrides["scrape_impersonate"] = args.impersonate
    if args.all_products:
        overrides["scrape_wines_only"] = False
    if args.browser_fallback:
        overrides["scrape_use_browser_fallback"] = True
    return get_settings().model_copy(update=overrides)


def _print_summary(report: ScrapeReport) -> None:
    print("\nScrape summary")
    print("--------------")
    print(f"  discovered      : {report.discovered}")
    print(
        f"  fetched         : {report.fetched} "
        f"({report.from_cache} from cache, {report.cache_hit_rate:.0%} hit rate)"
    )
    print(f"  written (wines) : {report.written}")
    print(f"  non-wine skipped: {report.non_wine_skipped}")
    print(f"  parse failures  : {report.parse_failures}")
    print(f"  fetch errors    : {report.fetch_errors}")
    print("  field coverage  :")
    for name, ratio in report.field_coverage.items():
        print(f"      {name:<16}: {ratio:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the Hedonism wines catalogue (I-1).")
    parser.add_argument("--limit", type=int, help="Cap the number of products processed.")
    parser.add_argument("--output", help="Output JSONL path.")
    parser.add_argument("--sitemap", help="Override the product sitemap URL.")
    parser.add_argument("--delay", type=float, help="Seconds to wait between requests.")
    parser.add_argument("--concurrency", type=int, help="Max concurrent requests.")
    parser.add_argument("--impersonate", help="curl_cffi browser profile (e.g. chrome131, safari).")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Rebuild the JSONL from cached HTML only; no network access.",
    )
    parser.add_argument(
        "--all-products",
        action="store_true",
        help="Keep every product, not just wines (disables the wines-only filter).",
    )
    parser.add_argument(
        "--browser-fallback",
        action="store_true",
        help="Render JS-only pages with Playwright when static HTML lacks markup.",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help="Human-readable logs instead of JSON.",
    )
    args = parser.parse_args()

    settings = _settings_from_args(args)
    configure_logging(settings.log_level, json_output=not args.log_console)
    report = run_from_cache(settings) if args.from_cache else asyncio.run(run_scrape(settings))
    _print_summary(report)


if __name__ == "__main__":
    main()
