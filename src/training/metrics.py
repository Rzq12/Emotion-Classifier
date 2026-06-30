"""Evaluation metrics for the emotion classifier.

F1-macro is the primary metric because the dataset is heavily imbalanced
(``anger`` is a sharp minority); accuracy alone would be misleading.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data.labels import LABELS


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Return accuracy, macro/weighted F1, and per-class F1.

    Per-class F1 keys are ``f1_<label>`` so they can be logged flat to MLflow.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    _, _, per_class_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    for label, score in zip(LABELS, per_class_f1, strict=True):
        metrics[f"f1_{label}"] = float(score)

    return metrics


def confusion_matrix_df(y_true: Sequence[int], y_pred: Sequence[int]):
    """Return the confusion matrix as a labeled pandas DataFrame.

    Rows = true label, columns = predicted label.
    """
    import pandas as pd

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
