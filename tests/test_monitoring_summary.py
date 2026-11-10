"""Unit tests for the live monitoring summary (prediction log -> dashboard numbers)."""

from __future__ import annotations

import pandas as pd
from src.monitoring.summary import MIN_SAMPLES_FOR_DRIFT, build_summary


def _reference() -> pd.DataFrame:
    labels = ["happiness"] * 60 + ["sadness"] * 30 + ["anger"] * 10
    return pd.DataFrame({"text": [f"review nomor {i}" for i in range(100)], "label": labels})


def _records(n: int, label: str = "happiness", day: str = "2026-11-01") -> list[dict]:
    return [
        {
            "timestamp": f"{day}T10:00:{i % 60:02d}+00:00",
            "text": f"review nomor {i}",
            "label": label,
            "confidence": 0.9,
        }
        for i in range(n)
    ]


def test_build_summary_empty_log_returns_zeroes_and_insufficient_data():
    summary = build_summary([], _reference())

    assert summary["n_predictions"] == 0
    assert summary["label_counts"] == {"anger": 0, "happiness": 0, "sadness": 0}
    assert summary["mean_confidence"] is None
    assert summary["low_confidence_share"] is None
    assert summary["daily"] == []
    assert summary["drift"]["status"] == "insufficient_data"
    assert summary["drift"]["min_samples"] == MIN_SAMPLES_FOR_DRIFT


def test_build_summary_below_min_samples_withholds_drift_verdict():
    summary = build_summary(_records(MIN_SAMPLES_FOR_DRIFT - 1), _reference())

    assert summary["drift"]["status"] == "insufficient_data"
    assert "label_psi" not in summary["drift"]


def test_build_summary_shifted_labels_reports_drift():
    summary = build_summary(_records(MIN_SAMPLES_FOR_DRIFT, label="anger"), _reference())

    assert summary["drift"]["status"] == "ok"
    assert summary["drift"]["label_verdict"] == "DRIFT"
    assert summary["label_distribution"]["anger"] == 1.0
    assert summary["reference_distribution"]["happiness"] == 0.6


def test_build_summary_matching_distribution_is_stable():
    records = (
        _records(60, "happiness") + _records(30, "sadness") + _records(10, "anger")
    )
    summary = build_summary(records, _reference())

    assert summary["drift"]["label_verdict"] == "STABLE"


def test_build_summary_missing_reference_reports_no_reference():
    summary = build_summary(_records(MIN_SAMPLES_FOR_DRIFT), None)

    assert summary["drift"]["status"] == "no_reference"
    assert summary["reference_distribution"] == {}


def test_build_summary_ignores_malformed_records():
    records = _records(3) + [{"label": "bogus", "text": "x"}, {"label": "anger"}, "not-a-dict"]

    assert build_summary(records, _reference())["n_predictions"] == 3


def test_build_summary_confidence_and_daily_counts():
    records = _records(2, day="2026-11-01") + _records(3, day="2026-11-02")
    records[0]["confidence"] = 0.5  # below the low-confidence threshold

    summary = build_summary(records, _reference())

    assert summary["low_confidence_share"] == 0.2
    assert summary["mean_confidence"] == 0.82
    assert summary["daily"] == [
        {"date": "2026-11-01", "count": 2},
        {"date": "2026-11-02", "count": 3},
    ]
    assert summary["first_timestamp"].startswith("2026-11-01")
    assert summary["last_timestamp"].startswith("2026-11-02")
