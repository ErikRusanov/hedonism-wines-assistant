"""Product-URL discovery via the catalogue's XML sitemap.

The listing pages are JavaScript-rendered (the static HTML only ships a handful
of featured products), so the cheap, complete and stable way to enumerate the
catalogue is ``sitemap-products.xml`` -- the path the site itself advertises in
``robots.txt``. The sitemap may be a flat ``<urlset>`` or a ``<sitemapindex>``
pointing at paged children; :func:`parse_sitemap` handles both and the async
:func:`discover_product_urls` follows any nesting.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Final

from hedonism_assistant.logging_config import get_logger

logger = get_logger(__name__)

_SITEMAP_NS: Final = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_PRODUCT_PATH: Final = "/product/"


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """Split one sitemap document into ``(product_urls, nested_sitemap_urls)``.

    A ``<urlset>`` yields product URLs (filtered to the ``/product/`` path); a
    ``<sitemapindex>`` yields nested sitemap URLs to recurse into. Namespaces are
    tolerated whether or not the document declares them.
    """
    root = ET.fromstring(xml_text)
    tag = root.tag.removeprefix(_SITEMAP_NS)

    locs = [
        (loc.text or "").strip()
        for loc in root.iter()
        if loc.tag.removeprefix(_SITEMAP_NS) == "loc" and loc.text
    ]

    if tag == "sitemapindex":
        return [], locs
    product_urls = [url for url in locs if _PRODUCT_PATH in url]
    return product_urls, []


async def discover_product_urls(
    fetch_text: Callable[[str], Awaitable[str]],
    sitemap_url: str,
    *,
    max_products: int | None = None,
) -> list[str]:
    """Return de-duplicated product URLs, recursing through nested sitemaps.

    ``fetch_text`` is injected (rather than importing the fetcher) so discovery
    stays decoupled and unit-testable. Order is preserved and duplicates removed;
    ``max_products`` caps the result for smoke runs.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    queue: deque[str] = deque((sitemap_url,))
    visited_sitemaps: set[str] = set()

    while queue:
        current = queue.popleft()
        if current in visited_sitemaps:
            continue
        visited_sitemaps.add(current)

        try:
            xml_text = await fetch_text(current)
        except Exception as exc:  # noqa: BLE001 - log and skip a bad sitemap leaf
            logger.warning("sitemap_fetch_failed", sitemap=current, error=str(exc))
            continue

        products, nested = parse_sitemap(xml_text)
        queue.extend(nested)
        for url in products:
            if url not in seen:
                seen.add(url)
                ordered.append(url)
                if max_products is not None and len(ordered) >= max_products:
                    logger.info("sitemap_discovery_capped", count=len(ordered), cap=max_products)
                    return ordered

    logger.info("sitemap_discovery_done", products=len(ordered), sitemaps=len(visited_sitemaps))
    return ordered
