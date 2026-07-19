"""Unit tests for prediction logging and drift check."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from src.monitoring.check_drift import (
    compute_drift_report,
    label_distribution,
    length_distribution,
    load_current,
    psi,
    render_markdown,
    simulate_drifted_sample,
)
from src.monitoring.prediction_log import PredictionLogger, read_predictions

# --- PredictionLogger -------------------------------------------------------


def test_prediction_logger_appends_jsonl(tmp_path):
    log_path = tmp_path / "sub" / "predictions.jsonl"
    logger = PredictionLogger(log_path)

    logger.log("aplikasi bagus", "happiness", 0.98)
    logger.log("error terus", "anger", 0.87654)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["text"] == "aplikasi bagus"
    assert first["label"] == "happiness"
    assert first["confidence"] == 0.98
    assert "timestamp" in first
    assert json.loads(lines[1])["confidence"] == 0.8765  # rounded to 4 decimals


def test_read_predictions_skips_malformed_lines(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(
        '{"text": "ok", "label": "happiness", "confidence": 0.9}\n'
        "not-json\n"
        '{"text": "gagal", "label": "anger", "confidence": 0.8}\n',
        encoding="utf-8",
    )
    records = read_predictions(log_path)
    assert [r["label"] for r in records] == ["happiness", "anger"]


def test_read_predictions_missing_file_returns_empty(tmp_path):
    assert read_predictions(tmp_path / "nope.jsonl") == []


# --- PSI & distributions ----------------------------------------------------


def test_psi_identical_distributions_is_zero():
    dist = {"anger": 0.1, "happiness": 0.6, "sadness": 0.3}
    assert psi(dist, dist) == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_flags_drift():
    reference = {"anger": 0.1, "happiness": 0.65, "sadness": 0.25}
    drifted = {"anger": 0.55, "happiness": 0.2, "sadness": 0.25}
    assert psi(reference, drifted) > 0.2


def test_label_distribution_covers_all_labels():
    dist = label_distribution(pd.Series(["happiness", "happiness", "sadness"]))
    assert dist["anger"] == 0.0
    assert dist["happiness"] == pytest.approx(2 / 3, abs=1e-3)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-3)


def test_length_distribution_sums_to_one():
    texts = pd.Series(["a" * 10, "b" * 60, "c" * 150, "d" * 300])
    dist = length_distribution(texts)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-3)
    assert dist["0-25"] == 0.25


# --- Drift report -----------------------------------------------------------


@pytest.fixture()
def reference_df():
    return pd.DataFrame(
        {
            "text": ["review bagus"] * 60 + ["review sedih"] * 30 + ["review marah"] * 10,
            "label": ["happiness"] * 60 + ["sadness"] * 30 + ["anger"] * 10,
        }
    )


def test_compute_drift_report_stable_when_same_data(reference_df):
    report = compute_drift_report(reference_df, reference_df)
    assert report["label_verdict"] == "STABLE"
    assert report["length_verdict"] == "STABLE"


def test_compute_drift_report_detects_simulated_drift(reference_df):
    current = simulate_drifted_sample(reference_df, n=200)
    report = compute_drift_report(reference_df, current)
    assert report["label_psi"] >= 0.2
    assert report["label_verdict"] == "DRIFT"
    assert report["confidence"]["mean"] > 0


def test_render_markdown_contains_sections(reference_df):
    current = simulate_drifted_sample(reference_df, n=100)
    md = render_markdown(compute_drift_report(reference_df, current))
    for heading in ("# Drift Report", "## Distribusi Label", "## Kesimpulan"):
        assert heading in md


def test_load_current_reads_jsonl(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    PredictionLogger(log_path).log("teks contoh", "sadness", 0.77)
    df = load_current(log_path)
    assert list(df["label"]) == ["sadness"]
    assert list(df["text"]) == ["teks contoh"]
