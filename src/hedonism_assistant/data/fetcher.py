"""Polite, cached, Cloudflare-aware HTTP fetching for the scraper.

The catalogue sits behind Cloudflare, which fingerprints the TLS/HTTP handshake
and returns ``403`` to ordinary HTTP clients *no matter what User-Agent they
send*. We therefore use :mod:`curl_cffi`, which impersonates a real browser's
fingerprint (``scrape_impersonate``) and sails past the passive check.

On top of that: concurrency is bounded, every request is followed by a
configurable delay, transient failures (timeouts, 403/429, 5xx) are retried with
exponential backoff, and fetched product HTML is cached to disk so re-running the
scrape is idempotent -- a second run touches the network only for unseen
products.

A Playwright fallback renders pages whose static HTML lacks the product markup
(or, more rarely, when Cloudflare serves an interactive JS challenge that
impersonation alone cannot clear). It is imported lazily and only used when
``scrape_use_browser_fallback`` is on, keeping the headless browser optional.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

import curl_cffi
from curl_cffi.requests import AsyncSession
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from hedonism_assistant.config import Settings
from hedonism_assistant.logging_config import get_logger

logger = get_logger(__name__)

# HTTP statuses worth retrying: anti-bot throttling (403/429) and upstream 5xx.
_RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}


@dataclass(slots=True, frozen=True)
class FetchResult:
    """Outcome of fetching one product page (immutable value object)."""

    url: str
    html: str
    from_cache: bool
    rendered: bool = False  # served via the Playwright fallback


class TransientHTTPError(Exception):
    """A retryable HTTP status (anti-bot throttling or upstream 5xx)."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"HTTP {status_code} for {url}")
        self.status_code = status_code
        self.url = url


def _is_retryable(exc: BaseException) -> bool:
    # curl_cffi raises CurlError for timeouts / connection resets / TLS issues.
    return isinstance(exc, TransientHTTPError | curl_cffi.CurlError)


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] or "index"
    # Keep cache filenames filesystem-safe without losing readability.
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)


class Fetcher:
    """Async HTTP client with browser impersonation, delays, retries and a cache."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._delay = settings.scrape_request_delay_seconds
        self._semaphore = asyncio.Semaphore(settings.scrape_max_concurrency)
        self._session = AsyncSession(**self._build_session_kwargs(settings))

    @staticmethod
    def _build_session_kwargs(settings: Settings) -> dict[str, object]:
        """Assemble the curl_cffi session config from settings.

        Static so the (subtle) impersonation-vs-plain header logic can be unit
        tested without constructing a live session.
        """
        kwargs: dict[str, object] = {
            "timeout": settings.scrape_timeout_seconds,
            "allow_redirects": True,
        }
        if settings.scrape_impersonate:
            # Impersonation supplies a full, coherent browser header set; we only
            # nudge the language. Overriding the UA here would break the profile.
            kwargs["impersonate"] = settings.scrape_impersonate
            kwargs["headers"] = {"Accept-Language": "en-GB,en;q=0.9"}
        else:
            kwargs["headers"] = {
                "User-Agent": settings.scrape_user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-GB,en;q=0.9",
            }
        return kwargs

    @cached_property
    def _html_cache(self) -> Path:
        """On-disk directory holding cached product HTML (computed once)."""
        return Path(self._settings.scrape_cache_dir) / "html"

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._session.close()

    async def fetch_text(self, url: str) -> str:
        """GET ``url`` and return its body text, with retries and a polite delay.

        Used for sitemap documents and as the static path for product pages.
        Not cached -- caching of product HTML is handled by
        :meth:`get_product_html`.
        """
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(self._settings.scrape_max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                reraise=True,
            ):
                with attempt:
                    text = await self._get(url)
            if self._delay:
                await asyncio.sleep(self._delay)
            return text

    async def _get(self, url: str) -> str:
        response = await self._session.get(url)
        status = response.status_code
        if status in _RETRYABLE_STATUS:
            raise TransientHTTPError(status, url)
        if status >= 400:
            raise RuntimeError(f"HTTP {status} for {url}")
        return response.text

    async def get_product_html(
        self,
        url: str,
        *,
        needs_render: Callable[[str], bool] | None = None,
    ) -> FetchResult:
        """Return product HTML, from cache when available, else from the network.

        ``needs_render`` decides whether the static HTML is missing the product
        markup; when it is and the browser fallback is enabled, the page is
        re-fetched through Playwright. The final HTML (static or rendered) is
        written to the cache so subsequent runs skip the network entirely.
        """
        cache_path = self._html_cache / f"{_slug_from_url(url)}.html"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return FetchResult(url=url, html=cache_path.read_text("utf-8"), from_cache=True)

        html = await self.fetch_text(url)
        rendered = False
        if needs_render is not None and needs_render(html):
            if self._settings.scrape_use_browser_fallback:
                logger.info("browser_fallback", url=url)
                html = await self.render_with_browser(url)
                rendered = True
            else:
                logger.warning("product_markup_missing", url=url, hint="enable browser fallback")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, "utf-8")
        return FetchResult(url=url, html=html, from_cache=False, rendered=rendered)

    async def render_with_browser(self, url: str) -> str:
        """Render ``url`` with headless Chromium and return the final HTML.

        Imported lazily so Playwright stays an optional dependency; raises a
        clear error if the ``scrape`` extra is not installed.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Playwright is required for the browser fallback. "
                'Install it with: uv pip install -e ".[scrape]" && playwright install chromium'
            ) from exc

        timeout_ms = int(self._settings.scrape_timeout_seconds * 1000)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page(user_agent=self._settings.scrape_user_agent)
                await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                return await page.content()
            finally:
                await browser.close()
