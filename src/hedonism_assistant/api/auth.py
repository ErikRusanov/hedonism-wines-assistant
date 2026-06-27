"""Shared-password access control for the chat UI and the API (I-9).

A single password (``settings.auth_password``) gates everything. When it is
empty auth is disabled — the default, so local dev and the test suite run
unguarded. When set, two callers are supported off the same secret:

* the **browser** logs in once at the branded login page, which POSTs to
  ``/login``; on success an HttpOnly cookie is set and the SPA loads. The cookie
  rides along with every subsequent same-origin request (``/chat`` SSE included);
* **programmatic** callers skip the cookie and present the password directly as
  ``Authorization: Bearer <password>`` or an ``X-Auth-Password`` header.

The guard is a pure-ASGI middleware (not :class:`BaseHTTPMiddleware`) so it never
wraps the response body — the ``/chat`` event stream passes through untouched.
``/health`` stays public for liveness probes, and ``OPTIONS`` preflights are let
through for CORS to answer.
"""

from __future__ import annotations

import base64
import secrets

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.types import ASGIApp, Receive, Scope, Send

from hedonism_assistant.config import Settings

# Open to everyone — liveness, and the login round-trip itself.
_PUBLIC_PATHS = frozenset({"/health", "/login"})


def _presented_secret(request: Request, cookie_name: str) -> str | None:
    """Pull the candidate password from cookie or header, in that order."""
    cookie = request.cookies.get(cookie_name)
    if cookie:
        return cookie

    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer ") :].strip()
    if header.startswith("Basic "):
        # Accept Basic auth too (any username); the password is what we check.
        try:
            decoded = base64.b64decode(header[len("Basic ") :]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
        return decoded.split(":", 1)[1] if ":" in decoded else decoded

    return request.headers.get("x-auth-password")


class AuthMiddleware:
    """Reject requests that lack the shared password (unless auth is disabled)."""

    def __init__(self, app: ASGIApp, password: str, cookie_name: str) -> None:
        self.app = app
        self._password = password
        self._cookie_name = cookie_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._password:
            await self.app(scope, receive, send)
            return

        if scope["method"] == "OPTIONS" or scope["path"] in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        presented = _presented_secret(request, self._cookie_name)
        if presented is not None and secrets.compare_digest(presented, self._password):
            await self.app(scope, receive, send)
            return

        # A browser navigation gets the login page; an API call gets clean JSON.
        accepts_html = scope["method"] == "GET" and "text/html" in request.headers.get("accept", "")
        if accepts_html:
            response: Response = HTMLResponse(_LOGIN_PAGE, status_code=status.HTTP_401_UNAUTHORIZED)
        else:
            response = JSONResponse(
                {"error": "unauthorized", "detail": "Authentication required."},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        await response(scope, receive, send)


router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    """Body of ``POST /login``."""

    password: str


@router.post("/login")
async def login(body: LoginRequest, request: Request) -> JSONResponse:
    """Validate the password and, on success, set the session cookie."""
    settings: Settings = request.app.state.settings
    if not settings.auth_password or not secrets.compare_digest(
        body.password, settings.auth_password
    ):
        return JSONResponse(
            {"error": "unauthorized", "detail": "Incorrect password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    response = JSONResponse({"ok": True})
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=settings.auth_password,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=60 * 60 * 24 * 30,  # 30 days
        path="/",
    )
    return response


# Self-contained login page, themed to match the chat UI. Served (with 401) for
# any unauthenticated browser navigation; posts to /login and reloads on success.
_LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hedonism Wines Assistant</title>
  <style>
    :root {
      --bg: #f6efe3; --panel: #fffdf8; --ink: #2c2118;
      --muted: #8c7257; --accent: #8a5a2b; --line: #e3d8c4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; height: 100vh; display: flex; align-items: center; justify-content: center;
      font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--ink);
    }
    .box {
      width: 100%; max-width: 360px; padding: 28px 26px; margin: 16px;
      background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
      box-shadow: 0 6px 30px #8a5a2b14;
    }
    h1 { margin: 0 0 4px; font-size: 19px; font-weight: 600; }
    p { margin: 0 0 18px; font-size: 13.5px; color: var(--muted); }
    form { display: flex; flex-direction: column; gap: 10px; }
    input {
      padding: 12px 14px; border-radius: 10px; border: 1px solid var(--line);
      background: var(--bg); color: var(--ink); font-size: 16px;
    }
    input:focus { outline: none; border-color: var(--accent); }
    button {
      padding: 12px; border: none; border-radius: 10px; cursor: pointer;
      background: var(--accent); color: var(--bg); font-weight: 600; font-size: 15px;
    }
    button:disabled { opacity: 0.5; cursor: default; }
    .error { min-height: 18px; font-size: 13px; color: #b3402e; }
  </style>
</head>
<body>
  <div class="box">
    <h1>Hedonism Wines Assistant</h1>
    <p>This cellar is private. Enter the password to continue.</p>
    <form id="form">
      <input id="password" type="password" autocomplete="current-password"
             placeholder="Password" autofocus />
      <div class="error" id="error"></div>
      <button id="submit" type="submit">Enter</button>
    </form>
  </div>
  <script>
    const form = document.getElementById("form");
    const password = document.getElementById("password");
    const submit = document.getElementById("submit");
    const error = document.getElementById("error");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      error.textContent = "";
      submit.disabled = true;
      try {
        const resp = await fetch("/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ password: password.value }),
        });
        if (resp.ok) {
          window.location.reload();
          return;
        }
        const body = await resp.json().catch(() => ({}));
        error.textContent = body.detail || "Incorrect password.";
      } catch (_) {
        error.textContent = "Connection error. Please try again.";
      } finally {
        submit.disabled = false;
        password.focus();
        password.select();
      }
    });
  </script>
</body>
</html>
"""
