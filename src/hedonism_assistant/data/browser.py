"""Fetching through a real browser (Playwright Chromium) -- the only fetch path.

The catalogue sits behind Cloudflare, which fingerprints the TLS/HTTP handshake
and 403s ordinary HTTP clients no matter the User-Agent or impersonation profile.
A genuine browser still gets through: it runs the full JS/TLS handshake and can
carry a ``cf_clearance`` cookie. This fetcher drives **one persistent Chromium
context** and reuses it across every request.

It caches fetched product HTML to disk (so re-runs touch the network only for
unseen products) and exposes ``fetch_text`` / ``get_product_html`` plus the
async-context-manager protocol, which is all the scraper and listing discovery
need.

Playwright is imported lazily and only when this fetcher is constructed, so it
stays an optional dependency (install with ``uv pip install -e ".[scrape]" &&
playwright install chromium``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urlparse

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hedonism_assistant.config import Settings
from hedonism_assistant.logging_config import get_logger

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext

logger = get_logger(__name__)

# HTTP statuses worth retrying: anti-bot throttling (403/429) and upstream 5xx.
_RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}


@dataclass(slots=True, frozen=True)
class FetchResult:
    """Outcome of fetching one product page (immutable value object)."""

    url: str
    html: str
    from_cache: bool


class TransientHTTPError(Exception):
    """A retryable HTTP status (anti-bot throttling or upstream 5xx)."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"HTTP {status_code} for {url}")
        self.status_code = status_code
        self.url = url


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] or "index"
    # Keep cache filenames filesystem-safe without losing readability.
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)


class BrowserFetcher:
    """Fetch pages with a persistent Playwright Chromium context."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._delay = settings.scrape_request_delay_seconds
        self._timeout_ms = int(settings.scrape_timeout_seconds * 1000)
        self._wait_until = settings.scrape_browser_wait_until
        self._semaphore = asyncio.Semaphore(settings.scrape_max_concurrency)
        self._html_cache = Path(settings.scrape_cache_dir) / "html"
        self._pw: Any = None
        self._browser: Any = None
        self._context: BrowserContext | None = None
        self._timeout_exc: type[BaseException] = TimeoutError

    async def __aenter__(self) -> Self:
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Playwright is required for scraping. Install it with: "
                'uv pip install -e ".[scrape]" && playwright install chromium'
            ) from exc

        self._timeout_exc = PlaywrightTimeoutError
        self._pw = await async_playwright().start()
        headless = self._settings.scrape_browser_headless
        user_data_dir = self._settings.scrape_browser_user_data_dir
        if user_data_dir:
            # Persistent profile: a cf_clearance cookie solved once survives.
            self._context = await self._pw.chromium.launch_persistent_context(
                user_data_dir,
                headless=headless,
                user_agent=self._settings.scrape_user_agent,
                locale="en-GB",
            )
        else:
            self._browser = await self._pw.chromium.launch(headless=headless)
            self._context = await self._browser.new_context(
                user_agent=self._settings.scrape_user_agent,
                locale="en-GB",
            )
        logger.info("browser_started", headless=headless, persistent=bool(user_data_dir))
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()

    async def fetch_text(self, url: str) -> str:
        """Render ``url`` and return its HTML, with retries and a polite delay."""
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(TransientHTTPError),
                stop=stop_after_attempt(self._settings.scrape_max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                reraise=True,
            ):
                with attempt:
                    html = await self._navigate(url)
            if self._delay:
                await asyncio.sleep(self._delay)
            return html

    async def _navigate(self, url: str) -> str:
        assert self._context is not None  # set in __aenter__
        page = await self._context.new_page()
        try:
            try:
                response = await page.goto(
                    url, wait_until=self._wait_until, timeout=self._timeout_ms
                )
            except self._timeout_exc as exc:
                raise TransientHTTPError(408, url) from exc
            status = response.status if response is not None else 0
            if status in _RETRYABLE_STATUS:
                raise TransientHTTPError(status, url)
            if status >= 400:
                raise RuntimeError(f"HTTP {status} for {url}")
            return await page.content()
        finally:
            await page.close()

    async def get_product_html(self, url: str) -> FetchResult:
        """Return product HTML from cache, else render it and cache the result."""
        cache_path = self._html_cache / f"{_slug_from_url(url)}.html"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return FetchResult(url=url, html=cache_path.read_text("utf-8"), from_cache=True)

        html = await self.fetch_text(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, "utf-8")
        return FetchResult(url=url, html=html, from_cache=False)
