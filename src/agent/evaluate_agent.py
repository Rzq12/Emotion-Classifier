"""Offline evaluation of the agent's routing decisions on the labeled test set.

The router is the agent's core decision, so it is measured directly: every
review in ``data/processed/test.csv`` is classified by the real IndoBERT model
and routed with the configured thresholds, then compared with the branch the
same policy would pick for the human (gold) label.

Metrics (no LLM call, so the run is free and reproducible):
- branch accuracy + gold-vs-predicted branch confusion matrix;
- escalation precision / recall (gold ``anger`` -> ``escalate``);
- missed-negative rate: gold negative reviews routed to ``archive`` — the
  costliest error, because nobody ever sees them;
- a threshold sweep over ``escalate_min_confidence`` to justify the chosen value;
- classify + route latency (p50 / p95) on this machine.

Run:
    python -m src.agent.evaluate_agent --config configs/agent.yaml
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.agent.config import AgentConfig
from src.agent.router import RouterConfig, route_by_emotion

BRANCHES: tuple[str, ...] = ("escalate", "draft", "archive")


@dataclass(frozen=True)
class Prediction:
    """One evaluated review: gold label, model label and model confidence."""

    gold: str
    label: str
    confidence: float


def expected_branch(gold_label: str, config: RouterConfig) -> str:
    """Branch the policy chooses when the classifier is certain of the gold label."""
    return route_by_emotion({"label": gold_label, "confidence": 1.0}, config)


def _predicted_branch(pred: Prediction, config: RouterConfig) -> str:
    return route_by_emotion({"label": pred.label, "confidence": pred.confidence}, config)


def evaluate_routing(preds: list[Prediction], config: RouterConfig) -> dict:
    """Compare predicted routing with gold-label routing.

    Args:
        preds: Classifier outputs paired with gold labels.
        config: Router thresholds under evaluation.

    Returns:
        Dict with accuracy, confusion counts, escalation precision/recall,
        missed-negative and false-escalation rates.

    Raises:
        ValueError: If ``preds`` is empty.
    """
    if not preds:
        raise ValueError("Tidak ada prediksi untuk dievaluasi.")

    confusion: Counter[tuple[str, str]] = Counter()
    missed_negative = 0
    for p in preds:
        pred_b = _predicted_branch(p, config)
        confusion[(expected_branch(p.gold, config), pred_b)] += 1
        if p.gold in config.draft_labels and pred_b == "archive":
            missed_negative += 1

    n = len(preds)
    correct = sum(confusion[(b, b)] for b in BRANCHES)
    pred_escalate = sum(confusion[(g, "escalate")] for g in BRANCHES)
    gold_escalate = sum(confusion[("escalate", p)] for p in BRANCHES)
    true_escalate = confusion[("escalate", "escalate")]
    gold_negative = sum(1 for p in preds if p.gold in config.draft_labels)

    return {
        "n": n,
        "branch_accuracy": _ratio(correct, n),
        "confusion": {f"{g}->{p}": confusion[(g, p)] for g in BRANCHES for p in BRANCHES},
        "escalation_precision": _ratio(true_escalate, pred_escalate),
        "escalation_recall": _ratio(true_escalate, gold_escalate),
        "escalations": pred_escalate,
        "false_escalation_rate": _ratio(pred_escalate - true_escalate, pred_escalate),
        "missed_negative_rate": _ratio(missed_negative, gold_negative),
        "missed_negatives": missed_negative,
    }


def sweep_thresholds(
    preds: list[Prediction], config: RouterConfig, thresholds: list[float]
) -> list[dict]:
    """Re-run :func:`evaluate_routing` for each ``escalate_min_confidence``."""
    rows = []
    for t in thresholds:
        m = evaluate_routing(preds, replace(config, escalate_min_confidence=float(t)))
        rows.append(
            {
                "threshold": float(t),
                "precision": m["escalation_precision"],
                "recall": m["escalation_recall"],
                "escalations": m["escalations"],
                "branch_accuracy": m["branch_accuracy"],
            }
        )
    return rows


def latency_stats(latencies_ms: list[float]) -> dict:
    """Return p50 / p95 / mean in milliseconds (zeros for an empty list)."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(latencies_ms)
    p95_idx = max(0, round(0.95 * len(ordered)) - 1)
    return {
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[p95_idx], 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


def classify_all(
    classifier, texts: list[str], golds: list[str], config: RouterConfig
) -> tuple[list[Prediction], list[float]]:
    """Run ``classifier.predict`` on each text, timing classify + route in ms."""
    preds, latencies = [], []
    for text, gold in zip(texts, golds, strict=True):
        start = time.perf_counter()
        out = classifier.predict(text)
        route_by_emotion(out, config)
        latencies.append((time.perf_counter() - start) * 1000)
        preds.append(Prediction(gold=gold, label=out["label"], confidence=float(out["confidence"])))
    return preds, latencies


def render_report(
    metrics: dict, sweep: list[dict], latency: dict, config: RouterConfig, source: str
) -> str:
    """Render the evaluation as a Markdown report."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "# Agent Routing Evaluation — Indo Review Intelligence",
        "",
        f"Generated: {now} | data: `{source}` (n={metrics['n']}) | "
        f"threshold eskalasi: **{config.escalate_min_confidence}**",
        "",
        "Classifier IndoBERT asli + router dari `configs/agent.yaml`, dibandingkan dengan "
        "cabang yang dipilih policy yang sama untuk label manusia (gold). Tanpa panggilan "
        "LLM — kualitas draft dinilai terpisah lewat approval rate di `/agent/stats`.",
        "",
        "## 1. Ringkasan",
        "",
        "| Metrik | Nilai |",
        "|---|---|",
        f"| Akurasi cabang | {metrics['branch_accuracy']:.3f} |",
        f"| Escalation precision | {metrics['escalation_precision']:.3f} |",
        f"| Escalation recall | {metrics['escalation_recall']:.3f} |",
        f"| False escalation rate | {metrics['false_escalation_rate']:.3f} |",
        f"| Missed-negative rate (negatif → archive) | {metrics['missed_negative_rate']:.3f} "
        f"({metrics['missed_negatives']} review) |",
        f"| Latency classify+route p50 / p95 | {latency['p50']} / {latency['p95']} ms |",
        "",
        "## 2. Confusion matrix cabang (baris = gold, kolom = prediksi)",
        "",
        "| gold \\ pred | " + " | ".join(BRANCHES) + " |",
        "|---|" + "---|" * len(BRANCHES),
    ]
    for g in BRANCHES:
        cells = " | ".join(str(metrics["confusion"][f"{g}->{p}"]) for p in BRANCHES)
        lines.append(f"| {g} | {cells} |")
    lines += [
        "",
        "## 3. Sweep threshold `escalate_min_confidence`",
        "",
        "| Threshold | Precision | Recall | Jumlah eskalasi | Akurasi cabang |",
        "|---|---|---|---|---|",
    ]
    for row in sweep:
        marker = " ←" if abs(row["threshold"] - config.escalate_min_confidence) < 1e-9 else ""
        lines.append(
            f"| {row['threshold']:.2f}{marker} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['escalations']} | {row['branch_accuracy']:.3f} |"
        )
    lines += [
        "",
        "## 4. Cara baca",
        "",
        "- **Missed-negative** adalah error paling mahal: review negatif yang diarsipkan tidak "
        "pernah dilihat manusia. Router sengaja mengirim label tak dikenal ke `draft`.",
        "- Threshold lebih rendah → recall eskalasi naik, tapi notifikasi palsu (precision) "
        "ikut naik. Pilih titik yang sesuai kapasitas tim yang menerima notifikasi.",
        "- Latency diukur di mesin lokal (CPU), tanpa LLM; `/agent/run` end-to-end "
        "menambah retrieval + generate draft.",
        "",
    ]
    return "\n".join(lines)


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def main() -> None:
    """CLI entry point: classify the test set, evaluate routing, write the report."""
    parser = argparse.ArgumentParser(description="Evaluate agent routing on the test set.")
    parser.add_argument("--config", default="configs/agent.yaml")
    parser.add_argument("--model-dir", default=None, help="Override MODEL_DIR.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N reviews.")
    args = parser.parse_args()

    from src.api.classifier import EmotionClassifier
    from src.api.config import get_settings

    agent_cfg = AgentConfig.load(args.config)
    with open(args.config, encoding="utf-8") as f:
        eval_cfg = (yaml.safe_load(f) or {}).get("evaluation", {})

    test_csv = eval_cfg.get("test_csv", "data/processed/test.csv")
    df = pd.read_csv(test_csv)
    if args.limit:
        df = df.head(args.limit)

    classifier = EmotionClassifier(model_dir=args.model_dir or get_settings().model_dir)
    preds, latencies = classify_all(
        classifier, df["text"].astype(str).tolist(), df["label"].tolist(), agent_cfg.router
    )

    metrics = evaluate_routing(preds, agent_cfg.router)
    sweep = sweep_thresholds(
        preds, agent_cfg.router, eval_cfg.get("threshold_sweep", [0.5, 0.75, 0.9])
    )
    report = render_report(metrics, sweep, latency_stats(latencies), agent_cfg.router, test_csv)

    out = Path(eval_cfg.get("report_path", "reports/agent_eval_report.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[evaluate_agent] report -> {out}")
    print(
        f"[evaluate_agent] accuracy={metrics['branch_accuracy']:.3f} "
        f"esc_p={metrics['escalation_precision']:.3f} esc_r={metrics['escalation_recall']:.3f} "
        f"missed_neg={metrics['missed_negative_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
