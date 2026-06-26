"""Reranking of retrieved candidates down to the final top-K (I-5).

The default reranker is an **LLM listwise** pass through the cheap utility model:
it sees the query and a compact numbered list of candidates and returns them
re-ordered by relevance. This adds no new service (the master plan's "minimum
services" constraint) and is a drop-in seam — Cohere/Voyage rerankers can
implement the same :class:`Reranker` protocol later.

Like query understanding, reranking must never fail a request: any chain/parse
error degrades to the input order, so a flaky utility model only costs ranking
quality, never an error.
"""

from __future__ import annotations

from typing import Final, Protocol

from openai.types.chat import ChatCompletionMessageParam

from hedonism_assistant.config import RerankerKind, Settings, get_settings
from hedonism_assistant.llm.json_output import loads_json
from hedonism_assistant.llm.openrouter import OpenRouterClient, get_openrouter_client
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.wine import RetrievedWine
from hedonism_assistant.vector_store.payload import normalize_critic_score

logger = get_logger(__name__)

# Cap how much of the (copyrighted) tasting note we feed the reranker per card —
# enough to judge style, bounded to keep the prompt compact.
_NOTE_SNIPPET_CHARS: Final = 200

# A single ranked entry from the model: candidate index plus optional relevance.
type RankEntry = tuple[int, float | None]


def _coerce_score(value: object) -> float | None:
    """Coerce a JSON relevance score to float; reject bools and non-numerics."""
    match value:
        case bool():
            return None
        case int() | float():
            return float(value)
        case _:
            return None


_SYSTEM_PROMPT = """\
You are a relevance reranker for a wine-catalogue search engine. Given the user's
query and a numbered list of candidate wines, order the candidates from most to
least relevant to the query.

Return ONLY a JSON object of this shape:
{"ranking": [{"index": int, "score": number}, ...]}
where "index" is the candidate's number from the list and "score" is a relevance
score in [0, 1]. List only the most relevant candidates, best first; you may omit
clearly irrelevant ones."""


class Reranker(Protocol):
    """Reorders retrieved candidates and truncates to ``top_k``."""

    async def rerank(
        self, query: str, candidates: list[RetrievedWine], *, top_k: int
    ) -> list[RetrievedWine]: ...


class NoOpReranker:
    """Passthrough reranker: keeps fusion order, just truncates to ``top_k``."""

    __slots__ = ()

    async def rerank(
        self, query: str, candidates: list[RetrievedWine], *, top_k: int
    ) -> list[RetrievedWine]:
        return candidates[:top_k]


class LLMListwiseReranker:
    """Listwise reranking via the cheap utility model (JSON mode)."""

    __slots__ = ("_client", "_settings")

    def __init__(self, client: OpenRouterClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def rerank(
        self, query: str, candidates: list[RetrievedWine], *, top_k: int
    ) -> list[RetrievedWine]:
        """Reorder ``candidates`` by model-judged relevance; never raises."""
        if len(candidates) <= 1:
            return candidates[:top_k]

        try:
            raw = await self._complete(query, candidates)
            ranking = self._parse_ranking(raw, len(candidates))
        except Exception as exc:  # noqa: BLE001 - resilience boundary: rerank must never fail a request
            logger.warning("rerank_failed", error=str(exc))
            return candidates[:top_k]

        if not ranking:
            return candidates[:top_k]

        ordered: list[RetrievedWine] = []
        ranked_indices: set[int] = set()
        for index, score in ranking:
            ranked_indices.add(index)
            ordered.append(candidates[index].model_copy(update={"rerank_score": score}))
        # Keep any candidates the model omitted, in their original order, so we
        # never silently drop a hit before truncation.
        ordered.extend(c for i, c in enumerate(candidates) if i not in ranked_indices)
        return ordered[:top_k]

    async def _complete(self, query: str, candidates: list[RetrievedWine]) -> str:
        """Call the utility model with the numbered candidate list."""
        listing = "\n".join(self._render(i, c) for i, c in enumerate(candidates))
        user = f"Query: {query}\n\nCandidates:\n{listing}"
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return await self._client.chat(
            messages,
            model=self._settings.utility_model,
            fallback_models=self._settings.utility_fallback_models,
            temperature=self._settings.rerank_temperature,
            response_format={"type": "json_object"},
        )

    @staticmethod
    def _render(index: int, candidate: RetrievedWine) -> str:
        """One compact line describing a candidate for the reranker prompt."""
        wine = candidate.wine
        location = "/".join(p for p in (wine.region, wine.sub_region) if p) or "—"
        grapes = ", ".join(wine.grapes) if wine.grapes else "—"
        score = max(
            (normalize_critic_score(s.score, s.scale) for s in wine.critic_scores), default=None
        )
        score_text = f", {score:.0f}/100" if score is not None else ""
        note = (wine.tasting_notes or "").strip().replace("\n", " ")
        if len(note) > _NOTE_SNIPPET_CHARS:
            note = note[:_NOTE_SNIPPET_CHARS].rstrip() + "…"
        note_text = f" {note}" if note else ""
        return (
            f"[{index}] {wine.name} — {location}, {grapes}, "
            f"£{wine.price:.0f}{score_text}.{note_text}"
        )

    @staticmethod
    def _parse_ranking(raw: str, n_candidates: int) -> list[RankEntry]:
        """Parse the model JSON into ``(index, score)`` pairs; drop bad entries."""
        payload = loads_json(raw)
        entries = payload.get("ranking") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []

        ordered: list[RankEntry] = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            # bool is an int subclass — exclude it so ``true`` isn't read as 1.
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            if not (0 <= index < n_candidates) or index in seen:
                continue
            seen.add(index)
            ordered.append((index, _coerce_score(entry.get("score"))))
        return ordered


def get_reranker(settings: Settings | None = None) -> Reranker:
    """Build the configured reranker."""
    settings = settings or get_settings()
    match settings.reranker_kind:
        case RerankerKind.LLM:
            return LLMListwiseReranker(get_openrouter_client(), settings)
        case RerankerKind.NONE:
            return NoOpReranker()
        # Future: case RerankerKind.COHERE / VOYAGE -> dedicated reranker.
