"""LLM-as-judge answer-quality metrics (I-8).

The lean RAGAS-style set the master plan calls for, scored by the cheap utility
model in JSON mode (the same model already used for query parsing and reranking —
no new service, the plan's "minimum services" constraint):

* **faithfulness** — the primary production guard against hallucination: of the
  claims the answer makes, how many are supported by the retrieved cards?
* **answer_relevancy** — how directly the answer addresses the question.

``context_precision``/``context_recall`` are deliberately left as future seams
(they need per-card verdicts / reference-answer attribution); the report keeps
``None`` placeholders for them.

Each metric is one ``chat`` call and **never raises**: any chain/parse error
degrades the metric to ``None`` (logged), exactly like :mod:`retrieval.rerank`,
so a flaky judge costs a data point, not a crashed harness.
"""

from __future__ import annotations

from typing import Final

from openai.types.chat import ChatCompletionMessageParam

from hedonism_assistant.config import Settings, get_settings
from hedonism_assistant.llm.json_output import loads_json
from hedonism_assistant.llm.openrouter import OpenRouterClient, get_openrouter_client
from hedonism_assistant.logging_config import get_logger
from hedonism_assistant.models.wine import RetrievedWine
from hedonism_assistant.vector_store.payload import normalize_critic_score

logger = get_logger(__name__)

# Floor for how much of the (copyrighted) note the judge sees; the effective cap
# is raised to ``generation_note_chars`` so the judge can see whatever the answer
# could have cited (see ``_render``).
_NOTE_SNIPPET_CHARS: Final = 200

_FAITHFULNESS_PROMPT: Final = """\
You are evaluating whether a wine assistant's answer is grounded in the catalogue
cards it was given. Extract the distinct factual claims the answer makes about
specific wines (names, producers, regions, vintages, prices, critic scores,
tasting characteristics). For each claim decide whether it is supported by the
cards. General wine knowledge not tied to a specific listed bottle counts as
supported.

Return ONLY a JSON object of this shape:
{"claims": [{"text": "...", "supported": true}, ...]}
If the answer makes no checkable claims, return {"claims": []}."""

_RELEVANCY_PROMPT: Final = """\
You are evaluating whether a wine assistant's answer addresses the user's
question. Judge only relevance — whether the answer responds to what was asked —
not factual accuracy or style.

Return ONLY a JSON object of this shape:
{"score": 0.0, "reason": "..."}
where "score" is in [0, 1]: 1.0 fully on-point, 0.0 entirely off-topic."""


class LLMJudge:
    """Score answer quality with the utility model; metrics degrade to ``None``."""

    __slots__ = ("_client", "_settings")

    def __init__(self, client: OpenRouterClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def faithfulness(
        self, question: str, answer: str, retrieved: list[RetrievedWine]
    ) -> float | None:
        """Fraction of the answer's claims supported by the cards, or ``None``."""
        cards = self._render_cards(retrieved)
        user = f"Question: {question}\n\nAnswer:\n{answer}\n\n<wines>\n{cards}\n</wines>"
        payload = await self._judge(_FAITHFULNESS_PROMPT, user)
        if not isinstance(payload, dict):
            return None
        claims = payload.get("claims")
        if not isinstance(claims, list):
            return None
        flags = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
        if not flags:
            # No checkable claims is vacuously faithful (e.g. a clarifying reply).
            return 1.0
        return sum(flags) / len(flags)

    async def answer_relevancy(self, question: str, answer: str) -> float | None:
        """Direct [0, 1] relevance of the answer to the question, or ``None``."""
        user = f"Question: {question}\n\nAnswer:\n{answer}"
        payload = await self._judge(_RELEVANCY_PROMPT, user)
        if not isinstance(payload, dict):
            return None
        return _clamp_unit(payload.get("score"))

    async def _judge(self, system: str, user: str) -> object | None:
        """One JSON-mode utility-model call; ``None`` on any chain/parse error."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            raw = await self._client.chat(
                messages,
                model=self._settings.utility_model,
                fallback_models=self._settings.utility_fallback_models,
                temperature=self._settings.eval_judge_temperature,
                response_format={"type": "json_object"},
            )
            return loads_json(raw)
        except Exception as exc:  # noqa: BLE001 - resilience boundary: judging must never crash the harness
            logger.warning("judge_failed", error=str(exc))
            return None

    def _render_cards(self, retrieved: list[RetrievedWine]) -> str:
        return "\n".join(self._render(i, c) for i, c in enumerate(retrieved, start=1))

    def _render(self, index: int, candidate: RetrievedWine) -> str:
        """One compact card line for the judge, mirroring the reranker's format."""
        wine = candidate.wine
        location = "/".join(p for p in (wine.region, wine.sub_region) if p) or "—"
        grapes = ", ".join(wine.grapes) if wine.grapes else "—"
        score = max(
            (normalize_critic_score(s.score, s.scale) for s in wine.critic_scores), default=None
        )
        score_text = f", {score:.0f}/100" if score is not None else ""
        note = (wine.tasting_notes or "").strip().replace("\n", " ")
        # Show the judge at least as much of the note as the generator could cite,
        # otherwise a claim grounded beyond the snippet reads as unsupported and
        # deflates faithfulness for no real fault.
        note_cap = max(_NOTE_SNIPPET_CHARS, self._settings.generation_note_chars)
        if len(note) > note_cap:
            note = note[:note_cap].rstrip() + "…"
        note_text = f" {note}" if note else ""
        return (
            f"[{index}] {wine.name} — {location}, {grapes}, "
            f"£{wine.price:.0f}{score_text}.{note_text}"
        )


def _clamp_unit(value: object) -> float | None:
    """Coerce a JSON number to a [0, 1] float; reject bools and non-numerics."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return max(0.0, min(1.0, float(value)))


def get_judge(settings: Settings | None = None) -> LLMJudge:
    """Build the judge from the shared client and settings."""
    settings = settings or get_settings()
    return LLMJudge(get_openrouter_client(), settings)
