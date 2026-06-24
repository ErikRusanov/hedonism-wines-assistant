"""Offline data track: scrape -> normalize -> index the Hedonism catalogue.

This package is import-light by design: nothing here is pulled in by the serving
API. The scraper (``scrape.py``) discovers product URLs from the sitemap,
fetches them politely with an on-disk cache, and parses each page into a
:class:`~hedonism_assistant.data.models.RawWine` record written to
``wines.raw.jsonl`` for the normalization stage (I-2) to consume.
"""
