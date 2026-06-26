"""Tolerant JSON extraction for chat-model responses.

Models reached through OpenRouter do not reliably honour the OpenAI
``response_format={"type": "json_object"}`` hint. Anthropic models in particular
ignore it and wrap their JSON in Markdown code fences (```json ... ```), and some
models prepend a sentence of prose. A plain ``json.loads`` on that raw content
fails with ``Expecting value: line 1 column 1``, so every utility-model call
would silently fall back. ``loads_json`` recovers the JSON from these shapes,
letting callers keep treating the utility model as a JSON endpoint.
"""

from __future__ import annotations

import json
import re

# Contents of the first ```/```json fenced block, if any.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def loads_json(raw: str) -> object:
    """Parse JSON from a model response, tolerating code fences and stray prose.

    Tries, in order: the string as-is; the contents of the first Markdown code
    fence; the substring spanning the first ``{`` to the last ``}``. Raises
    :class:`json.JSONDecodeError` if none yield valid JSON, so callers' existing
    ``except`` boundaries still fire on genuinely unparseable output.
    """
    candidates = [raw]
    if fence := _FENCE_RE.search(raw):
        candidates.append(fence.group(1))
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise last_error or json.JSONDecodeError("empty model response", raw or "", 0)


__all__ = ["loads_json"]
