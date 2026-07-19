"""Data drift check: reference dataset vs logged /classify predictions.

Compares label distribution, text-length distribution, and prediction
confidence between the reference data (train split) and current data (the
JSONL prediction log, or any CSV with ``text,label`` columns). Uses PSI
(Population Stability Index) — no external monitoring service needed
(PLAN Fase 6: manual script, report file as output).

Usage:
    python -m src.monitoring.check_drift
    python -m src.monitoring.check_drift --current data/monitoring/predictions.jsonl
    python -m src.monitoring.check_drift --simulate   # demo report with drifted sample

PSI rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, >= 0.2 significant drift.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.labels import LABELS
from src.monitoring.prediction_log import read_predictions

PSI_MODERATE = 0.1
PSI_DRIFT = 0.2
LOW_CONFIDENCE_THRESHOLD = 0.7
LENGTH_BINS = [0, 25, 50, 100, 200, 500, float("inf")]
LENGTH_BIN_LABELS = ["0-25", "26-50", "51-100", "101-200", "201-500", ">500"]


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = 1e-4) -> float:
    """Population Stability Index between two categorical distributions.

    Args:
        expected: Reference distribution (category -> proportion).
        actual: Current distribution (category -> proportion).
        eps: Floor applied to zero proportions to keep the log finite.
    """
    value = 0.0
    for key in set(expected) | set(actual):
        e = max(expected.get(key, 0.0), eps)
        a = max(actual.get(key, 0.0), eps)
        value += (a - e) * math.log(a / e)
    return round(value, 4)


def label_distribution(labels: pd.Series) -> dict[str, float]:
    """Proportion per emotion label, keyed by the canonical label set."""
    counts = labels.value_counts(normalize=True).to_dict()
    return {label: round(float(counts.get(label, 0.0)), 4) for label in LABELS}


def length_distribution(texts: pd.Series) -> dict[str, float]:
    """Proportion of texts per character-length bucket."""
    lengths = texts.astype(str).str.len()
    binned = pd.cut(lengths, bins=LENGTH_BINS, labels=LENGTH_BIN_LABELS, include_lowest=True)
    counts = binned.value_counts(normalize=True).to_dict()
    return {str(k): round(float(counts.get(k, 0.0)), 4) for k in LENGTH_BIN_LABELS}


def _verdict(value: float) -> str:
    if value >= PSI_DRIFT:
        return "DRIFT"
    if value >= PSI_MODERATE:
        return "MODERATE"
    return "STABLE"


def load_current(path: str | Path) -> pd.DataFrame:
    """Load current data from a JSONL prediction log or a CSV with text,label."""
    path = Path(path)
    if path.suffix == ".jsonl":
        records = read_predictions(path)
        return pd.DataFrame(records, columns=["timestamp", "text", "label", "confidence"])
    return pd.read_csv(path)


def simulate_drifted_sample(reference: pd.DataFrame, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build an intentionally drifted sample from the reference data (for demo).

    Oversamples the minority ``anger`` class hard so the label PSI crosses the
    drift threshold, and attaches synthetic low-ish confidences.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    weights = reference["label"].map({"anger": 8.0, "sadness": 2.0, "happiness": 0.5})
    sample = reference.sample(n=n, replace=True, weights=weights, random_state=seed).copy()
    sample["confidence"] = np.round(rng.uniform(0.45, 0.95, size=len(sample)), 4)
    return sample.reset_index(drop=True)


def compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compute all drift metrics; returns a plain dict (easy to render/test)."""
    ref_labels = label_distribution(reference["label"])
    cur_labels = label_distribution(current["label"])
    ref_lengths = length_distribution(reference["text"])
    cur_lengths = length_distribution(current["text"])

    label_psi = psi(ref_labels, cur_labels)
    length_psi = psi(ref_lengths, cur_lengths)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "label_distribution": {"reference": ref_labels, "current": cur_labels},
        "length_distribution": {"reference": ref_lengths, "current": cur_lengths},
        "label_psi": label_psi,
        "label_verdict": _verdict(label_psi),
        "length_psi": length_psi,
        "length_verdict": _verdict(length_psi),
    }

    if "confidence" in current.columns and current["confidence"].notna().any():
        conf = current["confidence"].dropna().astype(float)
        report["confidence"] = {
            "mean": round(float(conf.mean()), 4),
            "low_confidence_share": round(
                float((conf < LOW_CONFIDENCE_THRESHOLD).mean()), 4
            ),
            "threshold": LOW_CONFIDENCE_THRESHOLD,
        }
    return report


def render_markdown(report: dict) -> str:
    """Render the drift report dict as a markdown document."""
    lines = [
        "# Drift Report — Indo Review Intelligence",
        "",
        f"Generated: {report['generated_at']}",
        f"Reference: {report['n_reference']} rows | Current: {report['n_current']} rows",
        "",
        "## Ringkasan",
        "",
        "| Metrik | PSI | Verdict |",
        "|---|---|---|",
        f"| Distribusi label | {report['label_psi']} | {report['label_verdict']} |",
        f"| Distribusi panjang teks | {report['length_psi']} | {report['length_verdict']} |",
        "",
        f"Ambang PSI: < {PSI_MODERATE} stabil, {PSI_MODERATE}-{PSI_DRIFT} moderat, "
        f">= {PSI_DRIFT} drift signifikan.",
        "",
        "## Distribusi Label",
        "",
        "| Label | Reference | Current |",
        "|---|---|---|",
    ]
    ref_l = report["label_distribution"]["reference"]
    cur_l = report["label_distribution"]["current"]
    lines += [f"| {k} | {ref_l[k]:.2%} | {cur_l[k]:.2%} |" for k in ref_l]

    lines += [
        "",
        "## Distribusi Panjang Teks (karakter)",
        "",
        "| Bucket | Reference | Current |",
        "|---|---|---|",
    ]
    ref_n = report["length_distribution"]["reference"]
    cur_n = report["length_distribution"]["current"]
    lines += [f"| {k} | {ref_n[k]:.2%} | {cur_n[k]:.2%} |" for k in ref_n]

    if "confidence" in report:
        conf = report["confidence"]
        lines += [
            "",
            "## Confidence Prediksi",
            "",
            f"- Rata-rata confidence: **{conf['mean']:.4f}**",
            f"- Porsi prediksi low-confidence (< {conf['threshold']}): "
            f"**{conf['low_confidence_share']:.2%}**",
        ]

    verdicts = {report["label_verdict"], report["length_verdict"]}
    lines += [
        "",
        "## Kesimpulan",
        "",
        (
            "Terindikasi **drift signifikan** — pertimbangkan review data terbaru dan "
            "retraining model."
            if "DRIFT" in verdicts
            else (
                "Ada **pergeseran moderat** — pantau beberapa periode ke depan."
                if "MODERATE" in verdicts
                else "Distribusi **stabil** — tidak ada aksi yang diperlukan."
            )
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check data drift vs reference dataset.")
    parser.add_argument(
        "--reference", default="data/processed/train.csv", help="Reference CSV (text,label)."
    )
    parser.add_argument(
        "--current",
        default="data/monitoring/predictions.jsonl",
        help="Current data: JSONL prediction log or CSV (text,label).",
    )
    parser.add_argument(
        "--output", default="reports/drift_report.md", help="Markdown report output path."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Ignore --current; generate a drifted sample from reference (demo mode).",
    )
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)

    if args.simulate:
        current = simulate_drifted_sample(reference)
        print(f"[simulate] Using a synthetically drifted sample (n={len(current)}).")
    else:
        current = load_current(args.current)
        if current.empty:
            raise SystemExit(
                f"No current data found in '{args.current}'. "
                "Run some /classify requests first, or use --simulate for a demo report."
            )

    report = compute_drift_report(reference, current)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")

    print(f"Label PSI: {report['label_psi']} ({report['label_verdict']})")
    print(f"Length PSI: {report['length_psi']} ({report['length_verdict']})")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
