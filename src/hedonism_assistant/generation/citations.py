"""Derive grounded citations from the streamed answer text (I-6).

Generation and structured output pull in opposite directions: we want to stream
prose token-by-token, yet also report exactly which wines the answer leaned on.
Rather than pay for a second structured call, the generator is told to cite each
wine it mentions with its bracket number from the numbered context (``[2]``), and
this module recovers those references *after the fact* — a pure, deterministic
pass over the final text. No model, no network.

If the model emits no markers the citation list is empty by design; we do not
guess (e.g. "cite the whole context"), since an unmarked answer hasn't told us
which cards it actually used.
"""

from __future__ import annotations

import re
from typing import Final

from hedonism_assistant.models.chat import WineCitation
from hedonism_assistant.models.wine import RetrievedWine

# A 1-based bracket marker like ``[2]`` referencing a card in the numbered prompt.
_MARKER: Final = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, retrieved: list[RetrievedWine]) -> list[WineCitation]:
    """Map the ``[n]`` markers in ``answer`` to citations, in first-mention order.

    Markers are 1-based (matching how the prompt numbers cards). Out-of-range
    numbers are ignored and each wine is cited at most once.
    """
    citations: list[WineCitation] = []
    seen: set[int] = set()
    for match in _MARKER.finditer(answer):
        index = int(match.group(1))
        if not (1 <= index <= len(retrieved)) or index in seen:
            continue
        seen.add(index)
        citations.append(_to_citation(retrieved[index - 1]))
    return citations


def _to_citation(candidate: RetrievedWine) -> WineCitation:
    """Project a retrieved card onto the grounded citation contract."""
    wine = candidate.wine
    return WineCitation(
        wine_id=wine.id,
        name=wine.name,
        url=wine.url,
        price=wine.price,
        currency=wine.currency,
    )
