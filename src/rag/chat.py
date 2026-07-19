"""Chatbot: retrieve relevant reviews -> grounded LLM answer + sources.

Anti-hallucination (FR-3.2): the prompt instructs the model to answer only from
retrieved reviews and to say so explicitly when nothing relevant is found.
``sources`` lists the review IDs the answer actually cites (falling back to all
retrieved-relevant IDs when the model cites none). Answers are TTL-cached per
question+history+model to control LLM cost on repeated demo questions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from cachetools import TTLCache

from src.llm.base import LLMClient, LLMError
from src.llm.prompt_loader import render_prompt
from src.rag.formatting import format_reviews
from src.rag.retriever import HybridRetriever

NO_CONTEXT_MESSAGE = "Saya tidak menemukan review yang relevan dengan pertanyaan ini."
NO_HISTORY_TEXT = "(belum ada riwayat)"

# Bound the conversation context passed to the prompt.
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 300

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")


@dataclass
class ChatConfig:
    top_k: int = 8
    min_score: float = 0.55  # vector-similarity gate (calibrated via evaluate_retrieval)
    min_bm25: float = 1.0  # lexical relevance gate for vector-weak hits
    cache_ttl_seconds: int = 1800
    cache_max_size: int = 256


class ChatResponder:
    """Answer free-form questions grounded in retrieved reviews."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        config: ChatConfig | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.config = config or ChatConfig()
        self._cache: TTLCache = TTLCache(
            maxsize=self.config.cache_max_size,
            ttl=self.config.cache_ttl_seconds,
        )

    def _cache_key(self, question: str, history: list[dict]) -> str:
        llm_id = f"{getattr(self.llm, 'provider', '?')}:{getattr(self.llm, 'model', '?')}"
        history_part = "|".join(f"{m.get('role')}:{m.get('content')}" for m in history)
        raw = f"{llm_id}|{self.config.top_k}|{history_part}|{question}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        """Return ``{answer, sources[], cached}`` for a user question.

        Args:
            question: The current user question.
            history: Optional prior turns, each ``{"role": ..., "content": ...}``
                (most recent last). Only the tail is used, truncated per message.
        """
        if not question or not question.strip():
            return {"answer": NO_CONTEXT_MESSAGE, "sources": [], "cached": False}

        trimmed_history = _trim_history(history or [])
        key = self._cache_key(question.strip(), trimmed_history)
        if key in self._cache:
            return {**self._cache[key], "cached": True}

        reviews = self.retriever.query(question, n_results=self.config.top_k)
        relevant = [
            r
            for r in reviews
            if r.score >= self.config.min_score or r.bm25_score >= self.config.min_bm25
        ]

        if not relevant:
            return {"answer": NO_CONTEXT_MESSAGE, "sources": [], "cached": False}

        prompt = render_prompt(
            "chat.txt",
            history=_format_history(trimmed_history),
            reviews=format_reviews(relevant),
            question=question.strip(),
        )
        try:
            answer = self.llm.generate(prompt, temperature=0.2, max_tokens=512)
        except LLMError as exc:
            raise LLMError(f"Chat answer failed: {exc}") from exc

        declined = NO_CONTEXT_MESSAGE.rstrip(".") in answer
        result = {
            "answer": answer.strip(),
            "sources": [] if declined else _cited_sources(answer, relevant),
            "cached": False,
        }
        self._cache[key] = {k: v for k, v in result.items() if k != "cached"}
        return result


def _cited_sources(answer: str, relevant: list) -> list[str]:
    """IDs the answer actually cites; falls back to all relevant IDs if none."""
    retrieved_ids = [r.review_id for r in relevant]
    cited = set(_CITATION_RE.findall(answer)) & set(retrieved_ids)
    if cited:
        return [rid for rid in retrieved_ids if rid in cited]
    return retrieved_ids


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep the last few turns, each truncated, with only role+content."""
    tail = history[-MAX_HISTORY_MESSAGES:]
    return [
        {
            "role": str(m.get("role", "user")),
            "content": str(m.get("content", ""))[:MAX_HISTORY_CHARS],
        }
        for m in tail
        if str(m.get("content", "")).strip()
    ]


def _format_history(history: list[dict]) -> str:
    if not history:
        return NO_HISTORY_TEXT
    labels = {"user": "Pengguna", "bot": "Asisten", "assistant": "Asisten"}
    return "\n".join(f"{labels.get(m['role'], 'Pengguna')}: {m['content']}" for m in history)
