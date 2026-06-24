"""Offline data track: scrape -> normalize -> index the Hedonism catalogue.

This package is import-light by design: nothing here is pulled in by the serving
API. The scraper (``scrape.py``) discovers product URLs by paginating the
``/wines`` listing, fetches each page through a real browser (Playwright) with an
on-disk cache, and parses each page into a
:class:`~hedonism_assistant.data.models.RawWine` record written to
``wines.raw.jsonl``. The enrich stage (``enrich.py`` + ``normalize.py``, with the
optional LLM step in ``enricher.py``) turns those raw records into canonical
:class:`~hedonism_assistant.models.wine.Wine` cards in ``wines.enriched.jsonl``,
ready for indexing (I-3).
"""
