"""FastAPI application factory.

Wires configuration, structured logging, the chat/search/health endpoints, a
unified error format, CORS and the static chat page. On startup the catalogue
taxonomy is loaded from the live index and injected into the query parser; the
load degrades to a pass-through taxonomy if no index is reachable, so the app
always comes up.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hedonism_assistant import __version__
from hedonism_assistant.api import chat, health, search
from hedonism_assistant.api.errors import register_error_handlers
from hedonism_assistant.api.taxonomy import load_taxonomy
from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.logging_config import configure_logging, get_logger
from hedonism_assistant.retrieval.query_parser import get_query_parser
from hedonism_assistant.vector_store.client import get_wine_store

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, json_output=settings.log_json)
    logger.info("startup", version=__version__, collection=settings.qdrant_collection)

    taxonomy = await load_taxonomy(get_wine_store())
    app.state.taxonomy = taxonomy
    get_query_parser().set_taxonomy(taxonomy)

    yield
    logger.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Hedonism Wines Assistant",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(search.router)

    # Mounted last so it never shadows the API routes; ``html=True`` serves
    # index.html at ``/``.
    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app


app = create_app()
