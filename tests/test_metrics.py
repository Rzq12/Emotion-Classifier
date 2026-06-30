"""Unit tests for evaluation metrics."""

from __future__ import annotations

from src.data.labels import LABELS
from src.training.metrics import compute_metrics, confusion_matrix_df


def test_perfect_prediction_scores_one():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    for label in LABELS:
        assert m[f"f1_{label}"] == 1.0


def test_metrics_contain_per_class_keys():
    y_true = [0, 1, 2]
    y_pred = [0, 0, 2]
    m = compute_metrics(y_true, y_pred)
    expected_keys = {"accuracy", "f1_macro", "f1_weighted"} | {f"f1_{lbl}" for lbl in LABELS}
    assert expected_keys.issubset(m.keys())
    assert 0.0 <= m["f1_macro"] <= 1.0


def test_confusion_matrix_shape_and_labels():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 2, 1]
    cm = confusion_matrix_df(y_true, y_pred)
    assert cm.shape == (len(LABELS), len(LABELS))
    # Diagonal counts correct predictions; row sums equal true-class counts.
    assert cm.values.sum() == len(y_true)
    assert list(cm.index) == [f"true_{lbl}" for lbl in LABELS]
