"""Unit tests for label normalization and schema."""

from __future__ import annotations

import pytest
from src.data.labels import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    UnknownLabelError,
    normalize_label,
)


def test_schema_is_three_class_no_neutral():
    assert set(LABELS) == {"anger", "happiness", "sadness"}
    assert "neutral" not in LABELS


def test_label_maps_are_consistent():
    assert len(LABEL2ID) == len(ID2LABEL) == len(LABELS)
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HAPPINESS", "happiness"),
        ("happiness", "happiness"),
        ("  Anger ", "anger"),
        ("SADNESS", "sadness"),
        ("marah", "anger"),
        ("senang", "happiness"),
        ("sedih", "sadness"),
    ],
)
def test_normalize_label_casing_and_synonyms(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_rejects_unknown():
    with pytest.raises(UnknownLabelError):
        normalize_label("netral")
    with pytest.raises(UnknownLabelError):
        normalize_label("")


def test_normalize_label_rejects_non_string():
    with pytest.raises(UnknownLabelError):
        normalize_label(None)
