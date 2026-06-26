"""Build the catalogue taxonomy from the live Qdrant index at startup (I-7).

The query parser validates free-text filters (country/region/sub-region/grape)
against the values actually present in the catalogue. That taxonomy is loaded
here, once, from the indexed payloads and injected into the parser. Loading must
never block startup: an empty/unreachable index degrades to an empty taxonomy
(pass-through validation), exactly as the parser already handles before any data
is indexed.
"""

from __future__ import annotations

from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.wine import Wine
from hedonism_assistant.retrieval.taxonomy import Taxonomy
from hedonism_assistant.vector_store.client import QdrantWineStore

logger = get_logger(__name__)


async def load_taxonomy(store: QdrantWineStore) -> Taxonomy:
    """Build a :class:`Taxonomy` from the indexed wines; empty on any failure."""
    wines: list[Wine] = []
    try:
        async for payload in store.scroll_payloads():
            try:
                wines.append(Wine.model_validate(payload))
            except Exception as exc:  # noqa: BLE001 - skip a bad card, not the whole load
                logger.warning("taxonomy_payload_skipped", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - index empty/unreachable: degrade, never crash startup
        logger.warning("taxonomy_load_failed", error=str(exc))
        return Taxonomy()

    if not wines:
        logger.warning("taxonomy_empty")
        return Taxonomy()

    taxonomy = Taxonomy.from_wines(wines)
    logger.info(
        "taxonomy_loaded",
        wines=len(wines),
        countries=len(taxonomy.countries),
        regions=len(taxonomy.regions),
        sub_regions=len(taxonomy.sub_regions),
        grapes=len(taxonomy.grapes),
    )
    return taxonomy
