"""Unit tests for the training dataset loader."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.labels import LABEL2ID
from src.training.dataset import load_dataset


@pytest.fixture()
def tiny_csvs(tmp_path):
    """Create minimal train/val/test CSVs with canonical columns."""
    rows = pd.DataFrame(
        {"text": ["aplikasi bagus", "respon lama", "saya marah sekali"], "label": ["happiness", "sadness", "anger"]}
    )
    paths = {}
    for split in ("train", "val", "test"):
        p = tmp_path / f"{split}.csv"
        rows.to_csv(p, index=False)
        paths[split] = str(p)
    return paths


def test_load_dataset_encodes_labels(tiny_csvs):
    ds = load_dataset(tiny_csvs["train"], tiny_csvs["val"], tiny_csvs["test"])
    assert len(ds.train) == 3
    assert ds.train.labels == [LABEL2ID["happiness"], LABEL2ID["sadness"], LABEL2ID["anger"]]
    assert all(isinstance(t, str) for t in ds.train.texts)


def test_load_dataset_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["x"], "wrong": ["happiness"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_dataset(str(bad), str(bad), str(bad))
