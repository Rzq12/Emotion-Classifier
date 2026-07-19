"""Aggregate dataset statistics for the dashboard.

Computed from the processed CSVs (the data has no timestamps, so we report
emotion distribution and per-split counts rather than a fabricated time series).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yaml

from src.data.labels import LABELS


@lru_cache
def _data_cfg(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_stats(config_path: str = "configs/data.yaml") -> dict:
    """Return total counts, emotion distribution, and per-split breakdown."""
    cfg = _data_cfg(config_path)["processed"]
    splits = {"train": cfg["train_csv"], "val": cfg["val_csv"], "test": cfg["test_csv"]}

    by_split: dict[str, dict[str, int]] = {}
    by_emotion: dict[str, int] = {label: 0 for label in LABELS}
    total = 0

    for split_name, path in splits.items():
        counts = {label: 0 for label in LABELS}
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            by_split[split_name] = counts
            continue
        vc = df["label"].value_counts().to_dict()
        for label in LABELS:
            counts[label] = int(vc.get(label, 0))
            by_emotion[label] += counts[label]
            total += counts[label]
        by_split[split_name] = counts

    negative = by_emotion.get("anger", 0) + by_emotion.get("sadness", 0)
    return {
        "total": total,
        "by_emotion": by_emotion,
        "by_split": by_split,
        "negative_ratio": round(negative / total, 4) if total else 0.0,
    }
