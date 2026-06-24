"""Full-catalogue discovery by paginating the ``/wines`` listing.

The complete catalogue (~7.9k) is the paginated listing at ``/wines?pg=N``:

* the product grid is present in the page the browser renders,
* **0-indexed** (``pg=0`` is the first page),
* ~24 grid products per page, plus 3 constant promo tiles repeated on every page.

We walk pages until one yields no *new* product URLs, which terminates cleanly
whether the site ends pagination with an empty grid or by clamping to the last
page (both leave us with nothing new to add).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin

from hedonism_assistant.logging_config import get_logger

logger = get_logger(__name__)

_PRODUCT_RE = re.compile(r"/product/[a-z0-9][a-z0-9-]*")


def parse_listing_page(html: str, base_url: str) -> list[str]:
    """Return absolute, de-duplicated product URLs found on one listing page.

    Order-preserving and idempotent: a product linked several times (image,
    title, quick-view) collapses to a single URL.
    """
    seen: dict[str, None] = {}
    for match in _PRODUCT_RE.finditer(html):
        seen.setdefault(urljoin(base_url, match.group(0)), None)
    return list(seen)


async def discover_via_listing(
    fetch_text: Callable[[str], Awaitable[str]],
    base_url: str,
    *,
    listing_path: str = "/wines",
    max_products: int | None = None,
    max_pages: int = 1000,
) -> list[str]:
    """Enumerate catalogue product URLs by walking ``/wines?pg=0,1,2,...``.

    ``fetch_text`` is injected (rather than importing the fetcher) so discovery
    stays decoupled and unit-testable. Stops at the first page that contributes
    no new URLs; ``max_pages`` is a safety bound against an unexpected loop.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    for pg in range(max_pages):
        url = urljoin(base_url, f"{listing_path}?pg={pg}")
        try:
            html = await fetch_text(url)
        except Exception as exc:  # noqa: BLE001 - log and stop on a bad page
            logger.warning("listing_fetch_failed", page=pg, error=str(exc))
            break

        new = [u for u in parse_listing_page(html, base_url) if u not in seen]
        if not new:
            logger.info("listing_discovery_end", pages=pg, products=len(ordered))
            return ordered

        for product_url in new:
            seen.add(product_url)
            ordered.append(product_url)
            if max_products is not None and len(ordered) >= max_products:
                logger.info("listing_discovery_capped", products=len(ordered), cap=max_products)
                return ordered

        if pg % 25 == 0:
            logger.info("listing_discovery_progress", page=pg, products=len(ordered))

    logger.warning("listing_discovery_max_pages", max_pages=max_pages, products=len(ordered))
    return ordered
