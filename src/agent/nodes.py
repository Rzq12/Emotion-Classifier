"""Graph nodes: the units of work the agent performs on a review.

Each node is ``node(state, deps) -> dict`` returning only the fields it changes
(LangGraph merges the partial into ``ReviewState``). Nodes stay free of storage
and network side effects beyond the model/LLM calls that are their purpose —
persistence and notification are the API layer's job — which keeps them
testable with fake dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.config import DraftConfig
from src.agent.state import ReviewState
from src.llm.base import LLMClient, LLMError
from src.llm.prompt_loader import render_prompt
from src.rag.formatting import format_reviews

# Emotions whose reviews are worth retrieving as grounding context.
_NEGATIVE_EMOTIONS = ["anger", "sadness"]

_FALLBACK_DRAFT = (
    "Terima kasih atas masukan Anda. Kami mohon maaf atas ketidaknyamanan yang "
    "dialami dan tim kami akan segera menindaklanjuti laporan ini."
)


@dataclass
class AgentDeps:
    """Dependencies injected into nodes (real singletons or test fakes).

    ``classifier`` needs ``predict(text) -> {"label", "confidence"}``;
    ``retriever`` needs ``query(text, n_results, emotions) -> [RetrievedReview]``;
    ``llm`` is any :class:`~src.llm.base.LLMClient`.
    """

    classifier: object
    retriever: object
    llm: LLMClient
    draft_config: DraftConfig


def classify_node(state: ReviewState, deps: AgentDeps) -> dict:
    """Run the existing emotion classifier and record label + confidence."""
    result = deps.classifier.predict(state["review_text"])
    return {"label": result["label"], "confidence": float(result["confidence"])}


def escalate_node(state: ReviewState, deps: AgentDeps) -> dict:
    """High-priority negative review: draft an urgent, grounded reply."""
    draft, ids = _generate_grounded_draft(state, deps)
    return {
        "branch": "escalate",
        "priority": "high",
        "escalated": True,
        "draft": draft,
        "grounding_ids": ids,
    }


def draft_empathetic_node(state: ReviewState, deps: AgentDeps) -> dict:
    """Negative but non-urgent review: draft an empathetic reply for approval."""
    draft, ids = _generate_grounded_draft(state, deps)
    return {
        "branch": "draft",
        "priority": "normal",
        "escalated": False,
        "draft": draft,
        "grounding_ids": ids,
    }


def archive_node(state: ReviewState, deps: AgentDeps) -> dict:
    """Positive review: no reply needed, just archive it."""
    return {
        "branch": "archive",
        "priority": "none",
        "escalated": False,
        "draft": "",
        "grounding_ids": [],
    }


def _generate_grounded_draft(state: ReviewState, deps: AgentDeps) -> tuple[str, list[str]]:
    """Retrieve similar reviews, then have the LLM draft a reply grounded in them.

    Grounding first (FR anti-hallucination): the model only sees retrieved
    reviews as context. On any LLM failure we degrade to a safe canned reply
    rather than crashing the graph — a human approves it before it is sent.
    """
    cfg = deps.draft_config
    reviews = deps.retriever.query(
        state["review_text"],
        n_results=cfg.top_k,
        emotions=_NEGATIVE_EMOTIONS,
    )
    grounding_ids = [r.review_id for r in reviews]

    prompt = render_prompt(
        "agent_draft.txt",
        review=state["review_text"],
        emotion=state.get("label", "unknown"),
        similar_reviews=format_reviews(reviews),
    )
    try:
        draft = deps.llm.generate(
            prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        ).strip()
    except LLMError:
        draft = _FALLBACK_DRAFT
    return (draft or _FALLBACK_DRAFT, grounding_ids)
