"""Shared helpers for turning retrieved reviews into prompt context."""

from __future__ import annotations

from src.rag.vector_store import RetrievedReview


def format_reviews(reviews: list[RetrievedReview]) -> str:
    """Render retrieved reviews as a numbered, grounded context block."""
    if not reviews:
        return "(tidak ada review)"
    lines = [f"[{r.review_id}] ({r.emotion}) {r.text}" for r in reviews]
    return "\n".join(lines)
