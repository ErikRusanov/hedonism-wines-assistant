"""Smoke test for the application factory and healthcheck."""

from fastapi.testclient import TestClient

from hedonism_assistant import __version__
from hedonism_assistant.api.app import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
