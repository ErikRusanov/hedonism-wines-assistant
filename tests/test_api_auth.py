"""Auth gate tests for the shared-password access control (I-9).

The guard is exercised end to end via the ASGI stack: a configured
``auth_password`` must block unauthenticated calls, expose ``/health`` and
``/login`` only, and admit the browser (cookie) and programmatic callers
(Bearer / ``X-Auth-Password``) off the same secret. With no password set the
app is wide open, which the rest of the suite relies on.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hedonism_assistant.api.app import create_app
from hedonism_assistant.config import Settings
from hedonism_assistant.models.query import ParsedQuery, QueryIntent, WineFilters
from hedonism_assistant.retrieval.query_parser import get_query_parser
from hedonism_assistant.retrieval.retriever import get_retriever

_PASSWORD = "s3cret"


class _FakeParser:
    async def parse(self, message: str) -> ParsedQuery:
        return ParsedQuery(
            semantic_query=message, filters=WineFilters(), intent=QueryIntent.RECOMMENDATION
        )


class _FakeRetriever:
    async def retrieve(self, query: ParsedQuery) -> list:
        return []


def _guarded() -> TestClient:
    """A password-guarded app with the retrieval pipeline stubbed out, so the
    tests probe the auth gate alone — no model or Qdrant in the loop."""
    app = create_app(Settings(_env_file=None, auth_password=_PASSWORD))
    app.dependency_overrides[get_query_parser] = _FakeParser
    app.dependency_overrides[get_retriever] = _FakeRetriever
    return TestClient(app)


def test_disabled_when_password_empty() -> None:
    client = TestClient(create_app(Settings(_env_file=None)))
    # No auth configured: a normally-guarded path is reachable (404 from the
    # static mount, not 401 from the gate).
    assert client.get("/", headers={"accept": "text/html"}).status_code == 200


def test_health_and_login_are_public() -> None:
    client = _guarded()
    assert client.get("/health").status_code == 200
    # /login is reachable without a session (it is how you get one).
    assert client.post("/login", json={"password": "wrong"}).status_code == 401


def test_browser_navigation_without_auth_gets_login_page() -> None:
    resp = _guarded().get("/", headers={"accept": "text/html"})
    assert resp.status_code == 401
    assert "Enter the password" in resp.text


def test_api_without_auth_gets_json_401() -> None:
    resp = _guarded().post("/search", json={"query": "x"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_login_sets_cookie_and_unlocks_spa() -> None:
    client = _guarded()
    resp = client.post("/login", json={"password": _PASSWORD})
    assert resp.status_code == 200
    assert client.cookies.get("hw_auth") == _PASSWORD
    # The persisted cookie now serves the real SPA, not the login page.
    spa = client.get("/", headers={"accept": "text/html"})
    assert spa.status_code == 200
    assert "What can I pour you" in spa.text


def test_bearer_and_header_admit_programmatic_callers() -> None:
    client = _guarded()
    assert (
        client.post(
            "/search", json={"query": "x"}, headers={"Authorization": f"Bearer {_PASSWORD}"}
        ).status_code
        != 401
    )
    assert (
        client.post(
            "/search", json={"query": "x"}, headers={"X-Auth-Password": _PASSWORD}
        ).status_code
        != 401
    )


def test_wrong_password_stays_locked() -> None:
    client = _guarded()
    assert client.post("/login", json={"password": "nope"}).status_code == 401
    assert "hw_auth" not in client.cookies
    assert (
        client.post(
            "/search", json={"query": "x"}, headers={"Authorization": "Bearer nope"}
        ).status_code
        == 401
    )
