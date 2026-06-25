"""Offline data track: extract -> index the Hedonism catalogue.

This package is import-light by design: nothing here is pulled in by the serving
API. The catalogue is not scraped -- product-page HTML is captured by hand (see
``data/chrome_capture_prompt.md``) and saved as ``<slug>.html`` files. The
extract stage (``extract.py`` + ``parser.py`` + ``normalize.py``) parses each
file into a :class:`~hedonism_assistant.data.models.RawWine`, normalizes it into a
canonical :class:`~hedonism_assistant.models.wine.Wine` card, and writes
``wines.enriched.jsonl``. The index stage (``index.py``) embeds those cards
locally and upserts them into Qdrant (I-3).
"""
