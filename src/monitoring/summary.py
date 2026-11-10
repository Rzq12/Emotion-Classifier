"""Live monitoring summary built from the /classify prediction log.

Turns the JSONL log into the numbers the Monitoring tab shows: how many
predictions were served, their label mix, confidence health, daily volume, and
a PSI drift verdict against the training data. Drift is only computed once
enough predictions exist — a PSI over a handful of rows is noise, not signal.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.data.labels import LABELS
from src.monitoring.check_drift import (
    LOW_CONFIDENCE_THRESHOLD,
    compute_drift_report,
    label_distribution,
)

#: Below this many logged predictions the drift verdict is withheld.
MIN_SAMPLES_FOR_DRIFT = 30
#: Number of most recent active days reported in the daily volume series.
DAILY_WINDOW = 14


def _valid_records(records: list[dict]) -> list[dict]:
    """Keep only records usable for monitoring (known label, non-empty text)."""
    return [
        r
        for r in records
        if isinstance(r, dict) and r.get("label") in LABELS and str(r.get("text", "")).strip()
    ]


def _daily_counts(records: list[dict]) -> list[dict]:
    """Prediction volume per UTC date for the most recent active days."""
    counts = Counter(
        str(r["timestamp"])[:10] for r in records if len(str(r.get("timestamp", ""))) >= 10
    )
    return [{"date": day, "count": counts[day]} for day in sorted(counts)[-DAILY_WINDOW:]]


def build_summary(records: list[dict], reference: pd.DataFrame | None) -> dict:
    """Summarize logged predictions and compare them with the reference data.

    Args:
        records: Parsed prediction-log records (``timestamp, text, label, confidence``).
        reference: Training data with ``text`` and ``label`` columns, or ``None``
            when it is unavailable (drift status becomes ``no_reference``).

    Returns:
        A JSON-serializable dict matching ``MonitoringSummaryResponse``.
    """
    valid = _valid_records(records)
    n = len(valid)
    counts = Counter(r["label"] for r in valid)
    timestamps = sorted(str(r["timestamp"]) for r in valid if r.get("timestamp"))
    confidences = [float(r["confidence"]) for r in valid if r.get("confidence") is not None]

    summary: dict = {
        "n_predictions": n,
        "label_counts": {label: counts.get(label, 0) for label in LABELS},
        "label_distribution": {
            label: round(counts.get(label, 0) / n, 4) if n else 0.0 for label in LABELS
        },
        "reference_distribution": (
            label_distribution(reference["label"]) if reference is not None else {}
        ),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "low_confidence_share": (
            round(sum(c < LOW_CONFIDENCE_THRESHOLD for c in confidences) / len(confidences), 4)
            if confidences
            else None
        ),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "daily": _daily_counts(valid),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
    }

    drift: dict = {"status": "ok", "min_samples": MIN_SAMPLES_FOR_DRIFT}
    if reference is None:
        drift["status"] = "no_reference"
    elif n < MIN_SAMPLES_FOR_DRIFT:
        drift["status"] = "insufficient_data"
    else:
        report = compute_drift_report(reference, pd.DataFrame(valid))
        drift.update(
            label_psi=report["label_psi"],
            label_verdict=report["label_verdict"],
            length_psi=report["length_psi"],
            length_verdict=report["length_verdict"],
        )
    summary["drift"] = drift
    return summary
