"""Tests for full-catalogue discovery via listing pagination."""

from hedonism_assistant.data.listing import discover_via_listing, parse_listing_page

BASE = "https://hedonism.co.uk"

# Each page repeats 3 promo tiles and adds its own grid items, mirroring the site.
_PROMO = '<a href="/product/dom-perignon-2013"></a><a href="/product/ps250-gift-card"></a>'


def _page(*slugs: str) -> str:
    # Products are linked several times (image + title) to exercise de-duplication.
    tiles = "".join(
        f'<a href="/product/{s}"><img></a><a href="/product/{s}">{s}</a>' for s in slugs
    )
    return f"<html><body>{_PROMO}{tiles}</body></html>"


def test_parse_listing_page_dedupes_and_absolutises() -> None:
    urls = parse_listing_page(_page("chablis-droin-2023", "pontet-canet-2008"), BASE)
    assert urls == [
        "https://hedonism.co.uk/product/dom-perignon-2013",
        "https://hedonism.co.uk/product/ps250-gift-card",
        "https://hedonism.co.uk/product/chablis-droin-2023",
        "https://hedonism.co.uk/product/pontet-canet-2008",
    ]


async def test_discover_walks_until_no_new_products() -> None:
    pages = {
        "https://hedonism.co.uk/wines?pg=0": _page("a-2020", "b-2019"),
        "https://hedonism.co.uk/wines?pg=1": _page("c-2018", "d-2017"),
        # Page 2 is past the end: only the repeated promo tiles, nothing new.
        "https://hedonism.co.uk/wines?pg=2": _page(),
    }

    async def fetch_text(url: str) -> str:
        return pages[url]

    urls = await discover_via_listing(fetch_text, BASE)
    assert urls == [
        "https://hedonism.co.uk/product/dom-perignon-2013",
        "https://hedonism.co.uk/product/ps250-gift-card",
        "https://hedonism.co.uk/product/a-2020",
        "https://hedonism.co.uk/product/b-2019",
        "https://hedonism.co.uk/product/c-2018",
        "https://hedonism.co.uk/product/d-2017",
    ]


async def test_discover_respects_max_products() -> None:
    async def fetch_text(url: str) -> str:
        # Endless distinct pages; the cap must stop the walk.
        pg = url.rsplit("=", 1)[-1]
        return _page(f"wine-{pg}-x", f"wine-{pg}-y")

    urls = await discover_via_listing(fetch_text, BASE, max_products=5)
    assert len(urls) == 5
