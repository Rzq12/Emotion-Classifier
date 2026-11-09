"""Tests for the Fase 1 dataset preparation pipeline and RAG corpus loading."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml
from src.data.labels import UnknownLabelError
from src.data.prepare_dataset import clean_dataframe, prepare, split_train_val
from src.data.preprocessing import PreprocessConfig
from src.rag.corpus import load_reviews, review_id
from src.rag.embeddings import Embedder


def _raw_frame(n_per_class: int = 10) -> pd.DataFrame:
    rows = []
    for label, word in (("ANGER", "kesal"), ("happiness", "senang"), ("Sadness", "sedih")):
        for i in range(n_per_class):
            rows.append({"review_text": f"saya {word} sekali nomor {i}", "majority_vote": label})
    return pd.DataFrame(rows)


def test_clean_dataframe_normalizes_labels_and_columns():
    out = clean_dataframe(_raw_frame(2), "review_text", "majority_vote", PreprocessConfig())
    assert list(out.columns) == ["text", "label"]
    assert set(out["label"]) == {"anger", "happiness", "sadness"}


def test_clean_dataframe_drops_short_and_duplicate_rows():
    df = pd.DataFrame(
        {
            "review_text": ["aplikasi bagus", "aplikasi bagus", "ok", "😀😀"],
            "majority_vote": ["happiness"] * 4,
        }
    )
    out = clean_dataframe(df, "review_text", "majority_vote", PreprocessConfig(min_chars=3))
    assert out["text"].tolist() == ["aplikasi bagus"]


def test_clean_dataframe_unknown_label_raises():
    df = pd.DataFrame({"review_text": ["aplikasi bagus"], "majority_vote": ["neutral"]})
    with pytest.raises(UnknownLabelError):
        clean_dataframe(df, "review_text", "majority_vote", PreprocessConfig())


def test_split_train_val_stratified_keeps_every_class_in_val():
    df = clean_dataframe(_raw_frame(10), "review_text", "majority_vote", PreprocessConfig())
    train, val = split_train_val(df, val_size=0.3, random_state=42, stratify=True)
    assert len(train) + len(val) == len(df)
    assert set(val["label"]) == {"anger", "happiness", "sadness"}


def test_split_train_val_same_seed_is_reproducible():
    df = clean_dataframe(_raw_frame(10), "review_text", "majority_vote", PreprocessConfig())
    a, _ = split_train_val(df, val_size=0.3, random_state=7, stratify=True)
    b, _ = split_train_val(df, val_size=0.3, random_state=7, stratify=True)
    assert a["text"].tolist() == b["text"].tolist()


def test_prepare_end_to_end_writes_processed_csvs(tmp_path):
    raw_train, raw_test = tmp_path / "train_raw.csv", tmp_path / "test_raw.csv"
    _raw_frame(10).to_csv(raw_train, index=False)
    _raw_frame(3).to_csv(raw_test, index=False)
    out = tmp_path / "processed"
    cfg = {
        "raw": {
            "train_csv": str(raw_train),
            "test_csv": str(raw_test),
            "text_column": "review_text",
            "label_column": "majority_vote",
        },
        "processed": {
            "dir": str(out),
            "train_csv": str(out / "train.csv"),
            "val_csv": str(out / "val.csv"),
            "test_csv": str(out / "test.csv"),
        },
        "preprocessing": {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "reduce_repeated_chars": True,
            "normalize_slang": True,
            "emoji_strategy": "remove",
            "min_chars": 3,
        },
        "split": {"val_size": 0.2, "random_state": 42, "stratify": True},
    }
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = prepare(cfg_path)

    for split in ("train", "val", "test"):
        assert (out / f"{split}.csv").exists()
    assert len(result["train"]) + len(result["val"]) == 30
    assert len(result["test"]) == 9


# --- RAG corpus / embeddings --------------------------------------------------


def test_review_id_is_stable_and_split_prefixed():
    assert review_id("train", "halo") == review_id("train", "halo")
    assert review_id("train", "halo").startswith("train_")
    assert review_id("train", "halo") != review_id("test", "halo")


def test_load_reviews_merges_splits_and_dedupes_within_split(tmp_path):
    pd.DataFrame({"text": ["a b c", "a b c", "d e f"], "label": ["anger"] * 3}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"text": ["a b c"], "label": ["anger"]}).to_csv(tmp_path / "test.csv", index=False)
    df = load_reviews([str(tmp_path / "train.csv"), str(tmp_path / "test.csv")])
    assert len(df) == 3  # duplicate inside train dropped, same text in test kept
    assert set(df["split"]) == {"train", "test"}


class _FakeSentenceModel:
    def encode(self, texts, **kwargs):
        import numpy as np

        assert kwargs["normalize_embeddings"] is True
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_embedder_encode_uses_lazy_model_and_returns_lists():
    emb = Embedder(model_name="unused")
    emb._model = _FakeSentenceModel()
    assert emb.encode(["ab", "abcd"]) == [[2.0, 0.0], [4.0, 0.0]]
    assert emb.encode_one("abc") == [3.0, 0.0]
