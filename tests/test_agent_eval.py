"""Tests for the offline agent routing evaluation (pure functions, fake classifier)."""

from __future__ import annotations

import pytest
from src.agent.evaluate_agent import (
    Prediction,
    classify_all,
    evaluate_routing,
    expected_branch,
    latency_stats,
    render_report,
    sweep_thresholds,
)
from src.agent.router import RouterConfig

CFG = RouterConfig()


def test_expected_branch_maps_gold_labels_to_policy_branches():
    assert expected_branch("anger", CFG) == "escalate"
    assert expected_branch("sadness", CFG) == "draft"
    assert expected_branch("happiness", CFG) == "archive"


def test_evaluate_routing_perfect_predictions_score_one():
    preds = [
        Prediction("anger", "anger", 0.95),
        Prediction("sadness", "sadness", 0.9),
        Prediction("happiness", "happiness", 0.99),
    ]
    m = evaluate_routing(preds, CFG)
    assert m["branch_accuracy"] == 1.0
    assert m["escalation_precision"] == 1.0
    assert m["escalation_recall"] == 1.0
    assert m["missed_negative_rate"] == 0.0


def test_evaluate_routing_low_confidence_anger_counts_as_missed_escalation():
    m = evaluate_routing([Prediction("anger", "anger", 0.6)], CFG)
    assert m["confusion"]["escalate->draft"] == 1
    assert m["escalation_recall"] == 0.0
    assert m["missed_negative_rate"] == 0.0  # still seen by a human via draft


def test_evaluate_routing_negative_predicted_happy_is_missed_negative():
    preds = [Prediction("sadness", "happiness", 0.8), Prediction("sadness", "sadness", 0.8)]
    m = evaluate_routing(preds, CFG)
    assert m["missed_negatives"] == 1
    assert m["missed_negative_rate"] == 0.5


def test_evaluate_routing_false_escalation_rate():
    preds = [Prediction("sadness", "anger", 0.9), Prediction("anger", "anger", 0.9)]
    m = evaluate_routing(preds, CFG)
    assert m["escalations"] == 2
    assert m["false_escalation_rate"] == 0.5
    assert m["escalation_precision"] == 0.5


def test_evaluate_routing_empty_raises():
    with pytest.raises(ValueError):
        evaluate_routing([], CFG)


def test_sweep_thresholds_lower_threshold_never_reduces_recall():
    preds = [Prediction("anger", "anger", c) for c in (0.55, 0.7, 0.8, 0.95)]
    rows = sweep_thresholds(preds, CFG, [0.5, 0.75, 0.9])
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert rows[0]["recall"] == 1.0
    assert rows[-1]["escalations"] == 1


def test_latency_stats_percentiles_and_empty():
    stats = latency_stats([float(i) for i in range(1, 101)])
    assert stats["p50"] == 50.5
    assert stats["p95"] == 95.0
    assert latency_stats([]) == {"p50": 0.0, "p95": 0.0, "mean": 0.0}


class _FakeClassifier:
    def predict(self, text):
        return {"label": "anger" if "kesal" in text else "happiness", "confidence": 0.9}


def test_classify_all_pairs_predictions_with_gold_and_times_each():
    preds, lat = classify_all(_FakeClassifier(), ["kesal", "mantap"], ["anger", "sadness"], CFG)
    assert [p.label for p in preds] == ["anger", "happiness"]
    assert [p.gold for p in preds] == ["anger", "sadness"]
    assert len(lat) == 2 and all(x >= 0 for x in lat)


def test_render_report_contains_sections_and_marks_current_threshold():
    preds = [Prediction("anger", "anger", 0.9), Prediction("happiness", "happiness", 0.9)]
    m = evaluate_routing(preds, CFG)
    sweep = sweep_thresholds(preds, CFG, [0.5, 0.75])
    md = render_report(m, sweep, latency_stats([10.0, 20.0]), CFG, "test.csv")
    assert "## 2. Confusion matrix" in md
    assert "| 0.75 ← |" in md
    assert "Missed-negative" in md
