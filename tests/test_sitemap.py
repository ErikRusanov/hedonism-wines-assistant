"""Tests for product-URL discovery from XML sitemaps."""

from hedonism_assistant.data.sitemap import discover_product_urls, parse_sitemap

_NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'

URLSET = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset {_NS}>
  <url><loc>https://hedonism.co.uk/product/a-2020</loc></url>
  <url><loc>https://hedonism.co.uk/product/b-2019</loc></url>
  <url><loc>https://hedonism.co.uk/wines</loc></url>
</urlset>"""

INDEX = f"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex {_NS}>
  <sitemap><loc>https://hedonism.co.uk/sitemap-products.xml?page=1</loc></sitemap>
  <sitemap><loc>https://hedonism.co.uk/sitemap-products.xml?page=2</loc></sitemap>
</sitemapindex>"""


def test_parse_urlset_keeps_only_product_urls() -> None:
    products, nested = parse_sitemap(URLSET)
    assert products == [
        "https://hedonism.co.uk/product/a-2020",
        "https://hedonism.co.uk/product/b-2019",
    ]
    assert nested == []


def test_parse_index_returns_nested_sitemaps() -> None:
    products, nested = parse_sitemap(INDEX)
    assert products == []
    assert len(nested) == 2


async def test_discover_follows_index_and_dedupes() -> None:
    page1 = f"<urlset {_NS}><url><loc>https://hedonism.co.uk/product/a-2020</loc></url></urlset>"
    page2 = (
        f"<urlset {_NS}>"
        "<url><loc>https://hedonism.co.uk/product/a-2020</loc></url>"
        "<url><loc>https://hedonism.co.uk/product/c-2018</loc></url>"
        "</urlset>"
    )
    docs = {
        "root": INDEX,
        "https://hedonism.co.uk/sitemap-products.xml?page=1": page1,
        "https://hedonism.co.uk/sitemap-products.xml?page=2": page2,
    }

    async def fetch_text(url: str) -> str:
        return docs[url]

    urls = await discover_product_urls(fetch_text, "root")
    assert urls == [
        "https://hedonism.co.uk/product/a-2020",
        "https://hedonism.co.uk/product/c-2018",
    ]


async def test_discover_respects_max_products() -> None:
    async def fetch_text(url: str) -> str:
        return URLSET

    urls = await discover_product_urls(fetch_text, "root", max_products=1)
    assert urls == ["https://hedonism.co.uk/product/a-2020"]
