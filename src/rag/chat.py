"""Chatbot: retrieve relevant reviews -> grounded LLM answer + sources.

Anti-hallucination (FR-3.2): the prompt instructs the model to answer only from
retrieved reviews and to say so explicitly when nothing relevant is found.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.base import LLMClient, LLMError
from src.llm.prompt_loader import render_prompt
from src.rag.embeddings import Embedder
from src.rag.formatting import format_reviews
from src.rag.vector_store import ReviewVectorStore

NO_CONTEXT_MESSAGE = "Saya tidak menemukan review yang relevan dengan pertanyaan ini."


@dataclass
class ChatConfig:
    top_k: int = 8
    min_score: float = 0.15  # drop near-irrelevant hits below this similarity


class ChatResponder:
    """Answer free-form questions grounded in retrieved reviews."""

    def __init__(
        self,
        store: ReviewVectorStore,
        embedder: Embedder,
        llm: LLMClient,
        config: ChatConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.config = config or ChatConfig()

    def answer(self, question: str) -> dict:
        """Return ``{answer, sources[]}`` for a user question."""
        if not question or not question.strip():
            return {"answer": NO_CONTEXT_MESSAGE, "sources": []}

        query_vec = self.embedder.encode_one(question)
        reviews = self.store.query(query_embedding=query_vec, n_results=self.config.top_k)
        relevant = [r for r in reviews if r.score >= self.config.min_score]

        if not relevant:
            return {"answer": NO_CONTEXT_MESSAGE, "sources": []}

        prompt = render_prompt(
            "chat.txt",
            reviews=format_reviews(relevant),
            question=question.strip(),
        )
        try:
            answer = self.llm.generate(prompt, temperature=0.2, max_tokens=512)
        except LLMError as exc:
            raise LLMError(f"Chat answer failed: {exc}") from exc

        return {
            "answer": answer.strip(),
            "sources": [r.review_id for r in relevant],
        }
