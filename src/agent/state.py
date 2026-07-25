"""Shared state passed between graph nodes.

``ReviewState`` is the single object that flows through the LangGraph state
machine. ``total=False`` because early nodes populate fields incrementally
(``classify`` sets ``label``/``confidence`` before ``draft`` sets ``draft``).
"""

from __future__ import annotations

from typing import Literal, TypedDict

# The branch a review is routed to after classification.
Branch = Literal["escalate", "draft", "archive"]


class ReviewState(TypedDict, total=False):
    """State threaded through the agent graph for one review."""

    # --- input ---
    review_text: str

    # --- set by classify node ---
    label: str
    confidence: float

    # --- set by router (conditional edge) ---
    branch: Branch

    # --- set by escalate / draft / archive nodes ---
    priority: Literal["high", "normal", "none"]
    draft: str
    grounding_ids: list[str]
    escalated: bool
