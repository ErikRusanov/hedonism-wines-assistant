"""FastAPI application factory.

Iteration 0 wires up configuration, structured logging and a healthcheck. The
chat/search endpoints land in later iterations (see .claude/plans).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from hedonism_assistant import __version__
from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``."""

    status: Literal["ok"] = "ok"
    version: str


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, json_output=settings.log_json)
    logger.info("startup", version=__version__, collection=settings.qdrant_collection)
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

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    return app


app = create_app()
