"""Emotion label schema and normalization.

The raw files use inconsistent casing (train: ``HAPPINESS``, test: ``happiness``).
This module defines the canonical 3-class emotion schema (no neutral, matching the
actual dataset) and the maps needed for model training/serving.
"""

from __future__ import annotations

# Canonical label order. Index position == class id used during training.
LABELS: tuple[str, ...] = ("anger", "happiness", "sadness")

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

# Accept known surface forms / synonyms and map to canonical labels.
_ALIASES: dict[str, str] = {
    "anger": "anger",
    "angry": "anger",
    "marah": "anger",
    "happiness": "happiness",
    "happy": "happiness",
    "joy": "happiness",
    "senang": "happiness",
    "sadness": "sadness",
    "sad": "sadness",
    "sedih": "sadness",
}


class UnknownLabelError(ValueError):
    """Raised when a label cannot be mapped to the canonical schema."""


def normalize_label(label: str) -> str:
    """Map a raw label (any casing/known synonym) to a canonical emotion label.

    Raises :class:`UnknownLabelError` for unrecognized values so bad data fails
    loudly during preparation instead of silently corrupting the training set.
    """
    if not isinstance(label, str):
        raise UnknownLabelError(f"Label must be a string, got {type(label)!r}")

    key = label.strip().lower()
    if key not in _ALIASES:
        raise UnknownLabelError(f"Unrecognized label: {label!r}")
    return _ALIASES[key]
