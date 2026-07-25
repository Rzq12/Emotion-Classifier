"""Conditional routing: decide a review's branch from emotion + confidence.

This is the core agentic decision — not a passive label, but a policy that
weighs the predicted emotion against a confidence threshold to choose an
action. Kept as a pure function (no I/O) so its edge cases are unit-testable
(confidence exactly at threshold, unknown label, empty text).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.state import Branch, ReviewState


@dataclass(frozen=True)
class RouterConfig:
    """Thresholds that steer routing (loaded from ``configs/agent.yaml``)."""

    escalate_label: str = "anger"
    escalate_min_confidence: float = 0.75
    draft_labels: tuple[str, ...] = ("anger", "sadness")
    archive_labels: tuple[str, ...] = ("happiness",)

    @classmethod
    def from_dict(cls, cfg: dict) -> RouterConfig:
        return cls(
            escalate_label=cfg.get("escalate_label", "anger"),
            escalate_min_confidence=float(cfg.get("escalate_min_confidence", 0.75)),
            draft_labels=tuple(cfg.get("draft_labels", ["anger", "sadness"])),
            archive_labels=tuple(cfg.get("archive_labels", ["happiness"])),
        )


def route_by_emotion(state: ReviewState, config: RouterConfig | None = None) -> Branch:
    """Return the branch (``escalate`` | ``draft`` | ``archive``) for ``state``.

    Policy:
    - ``escalate`` — urgent negative signal: the escalate label (anger) at or
      above the confidence threshold. Needs human attention + notification.
    - ``draft`` — any negative emotion below the escalate bar: worth an
      empathetic reply drafted for approval. Ambiguous/unknown labels also land
      here so a human always sees them (fail toward review, not silence).
    - ``archive`` — positive emotion: no action, just recorded.

    Args:
        state: Review state; must carry ``label`` and ``confidence``.
        config: Routing thresholds. Defaults to :class:`RouterConfig` defaults.

    Returns:
        The chosen branch name.
    """
    config = config or RouterConfig()
    label = str(state.get("label", "")).strip().lower()
    confidence = float(state.get("confidence", 0.0))

    if label == config.escalate_label and confidence >= config.escalate_min_confidence:
        return "escalate"
    if label in config.draft_labels:
        return "draft"
    if label in config.archive_labels:
        return "archive"
    # Unknown / unmapped label: route to draft so a human reviews it.
    return "draft"
