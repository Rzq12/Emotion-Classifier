"""Insight generation: retrieve negative reviews -> LLM -> structured summary.

Grounded (FR-2.3): the LLM only sees retrieved reviews, and the result is cached
per query to control LLM cost (FR-2.4). ``example_review_ids`` returned by the
LLM are validated against the retrieved set so hallucinated IDs never surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from cachetools import TTLCache

from src.llm.base import LLMClient, LLMError
from src.llm.prompt_loader import render_prompt
from src.rag.formatting import format_reviews
from src.rag.retriever import HybridRetriever

# Minimum reviews required before attempting a meaningful insight (UIUX_FLOW 3.2).
MIN_REVIEWS_FOR_INSIGHT = 5

SAMPLE_NOTE = (
    "Statistik tema dihitung dari {n} review paling relevan dengan fokus ini, "
    "bukan dari seluruh korpus."
)


@dataclass
class InsightConfig:
    negative_emotions: list[str] = field(default_factory=lambda: ["anger", "sadness"])
    top_k: int = 20
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 128


class InsightGenerator:
    """Retrieve negative reviews for a query and summarize them via an LLM."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        config: InsightConfig | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.config = config or InsightConfig()
        self._cache: TTLCache = TTLCache(
            maxsize=self.config.cache_max_size,
            ttl=self.config.cache_ttl_seconds,
        )

    def _cache_key(self, query: str) -> str:
        # Include provider/model so switching LLMs never serves stale results.
        llm_id = f"{getattr(self.llm, 'provider', '?')}:{getattr(self.llm, 'model', '?')}"
        raw = f"{llm_id}|{query}|{sorted(self.config.negative_emotions)}|{self.config.top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, query: str = "keluhan utama pengguna", use_cache: bool = True) -> dict:
        """Return a structured insight dict for the given focus query.

        Result shape: ``{summary, themes[], sample_quotes[], recommendations[],
        note, cached, n_reviews}``.
        """
        key = self._cache_key(query)
        if use_cache and key in self._cache:
            return {**self._cache[key], "cached": True}

        reviews = self.retriever.query(
            query,
            n_results=self.config.top_k,
            emotions=self.config.negative_emotions,
        )

        if len(reviews) < MIN_REVIEWS_FOR_INSIGHT:
            return {
                "summary": "Data review negatif belum cukup untuk insight yang bermakna.",
                "themes": [],
                "sample_quotes": [],
                "recommendations": [],
                "note": "",
                "cached": False,
                "n_reviews": len(reviews),
            }

        prompt = render_prompt("insight.txt", reviews=format_reviews(reviews))
        try:
            raw = self.llm.generate(prompt, temperature=0.2, max_tokens=1024)
        except LLMError as exc:
            raise LLMError(f"Insight generation failed: {exc}") from exc

        result = _parse_insight_json(raw)
        _validate_example_ids(result, {r.review_id for r in reviews})
        result["note"] = SAMPLE_NOTE.format(n=len(reviews))
        result["cached"] = False
        result["n_reviews"] = len(reviews)

        if use_cache:
            self._cache[key] = {k: v for k, v in result.items() if k != "cached"}
        return result


def _validate_example_ids(result: dict, retrieved_ids: set[str]) -> None:
    """Drop hallucinated review IDs the LLM was never shown (in place)."""
    for theme in result.get("themes", []):
        if isinstance(theme, dict):
            ids = theme.get("example_review_ids", [])
            theme["example_review_ids"] = [i for i in ids if i in retrieved_ids]


def _parse_insight_json(raw: str) -> dict:
    """Parse LLM output as JSON, tolerating markdown fences/surrounding text."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Graceful fallback: surface the raw text rather than crashing.
        return {
            "summary": raw.strip()[:500],
            "themes": [],
            "sample_quotes": [],
            "recommendations": [],
        }

    data.setdefault("summary", "")
    data.setdefault("themes", [])
    data.setdefault("sample_quotes", [])
    data.setdefault("recommendations", [])
    return data
