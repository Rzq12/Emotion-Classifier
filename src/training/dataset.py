"""Load the processed emotion dataset for training/evaluation.

Reads the CSVs produced by ``src.data.prepare_dataset`` (columns ``text,label``)
and exposes them as plain lists plus integer-encoded labels using the canonical
schema in ``src.data.labels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.labels import LABEL2ID

TEXT_COL = "text"
LABEL_COL = "label"


@dataclass(frozen=True)
class Split:
    """A single dataset split."""

    texts: list[str]
    labels: list[int]  # integer-encoded via LABEL2ID

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class EmotionDataset:
    """Train / val / test splits ready for modeling."""

    train: Split
    val: Split
    test: Split


def _load_split(csv_path: str | Path) -> Split:
    df = pd.read_csv(csv_path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    texts = df[TEXT_COL].astype(str).tolist()
    labels = [LABEL2ID[label] for label in df[LABEL_COL]]
    return Split(texts=texts, labels=labels)


def load_dataset(
    train_csv: str | Path = "data/processed/train.csv",
    val_csv: str | Path = "data/processed/val.csv",
    test_csv: str | Path = "data/processed/test.csv",
) -> EmotionDataset:
    """Load all three processed splits."""
    return EmotionDataset(
        train=_load_split(train_csv),
        val=_load_split(val_csv),
        test=_load_split(test_csv),
    )
