"""Tests for MLflow helpers, run registry and the TF-IDF baseline (local SQLite only)."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
import yaml
from src.tracking import registry
from src.tracking.mlflow_utils import _save_heatmap, setup_tracking
from src.training import baseline


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite store inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield tmp_path
    mlflow.end_run()


def _tiny_split_csvs(tmp_path) -> dict[str, str]:
    texts = {
        "anger": ["kesal sekali error terus", "marah aplikasi lemot parah", "kesal transaksi gagal"],
        "happiness": ["senang aplikasi bagus", "mantap pelayanan cepat", "senang driver ramah"],
        "sadness": ["sedih pesanan hilang", "kecewa dokter tidak datang", "sedih refund lama"],
    }
    rows = [{"text": t, "label": lbl} for lbl, ts in texts.items() for t in ts]
    df = pd.DataFrame(rows * 2)
    paths = {}
    for split in ("train", "val", "test"):
        p = tmp_path / f"{split}.csv"
        df.to_csv(p, index=False)
        paths[split] = str(p)
    return paths


def test_save_heatmap_writes_png(tmp_path):
    cm = pd.DataFrame([[3, 1], [0, 4]], index=["a", "b"], columns=["a", "b"])
    out = tmp_path / "cm.png"
    _save_heatmap(cm, out, title="cm")
    assert out.stat().st_size > 0


def test_list_runs_new_experiment_returns_empty_table(tracking):
    # setup_tracking() -> mlflow.set_experiment() creates a missing experiment,
    # so an unknown name yields an empty table rather than an error.
    assert registry.list_runs("does-not-exist").empty


def test_list_runs_sorted_by_selection_metric_desc(tracking):
    setup_tracking("exp-sort")
    for name, score in (("low", 0.5), ("high", 0.9), ("mid", 0.7)):
        with mlflow.start_run(run_name=name):
            mlflow.log_metric("test_f1_macro", score)
    table = registry.list_runs("exp-sort")
    assert table["tags.mlflow.runName"].tolist() == ["high", "mid", "low"]


def test_register_best_registers_top_run_and_tags_it(tracking, monkeypatch):
    setup_tracking("exp-reg")
    for score in (0.6, 0.8):
        with mlflow.start_run():
            mlflow.log_metric("test_f1_macro", score)
    best_id = registry.list_runs("exp-reg")["run_id"].iloc[0]

    register = MagicMock(return_value=MagicMock(version="3"))
    client = MagicMock()
    monkeypatch.setattr(registry.mlflow, "register_model", register)
    monkeypatch.setattr(registry.mlflow.tracking, "MlflowClient", lambda: client)

    assert registry.register_best("exp-reg") == "3"
    register.assert_called_once_with(
        model_uri=f"runs:/{best_id}/model", name=registry.REGISTERED_MODEL_NAME
    )
    client.set_model_version_tag.assert_called_once_with(
        registry.REGISTERED_MODEL_NAME, "3", "stage", "production-candidate"
    )


def test_register_best_no_runs_returns_none(tracking):
    setup_tracking("exp-empty")
    assert registry.register_best("exp-empty") is None


def test_build_pipeline_unknown_classifier_raises():
    cfg = {
        "baseline": {
            "word_ngram_range": [1, 1],
            "min_df": 1,
            "max_features": 100,
            "classifier": "xgboost",
            "C": 1.0,
            "max_iter": 100,
            "class_weight_balanced": True,
        }
    }
    with pytest.raises(ValueError, match="Unknown baseline classifier"):
        baseline.build_pipeline(cfg)


@pytest.mark.parametrize("classifier", ["logreg", "linearsvc"])
def test_baseline_run_logs_metrics_to_mlflow(tracking, classifier):
    paths = _tiny_split_csvs(tracking)
    cfg = {
        "data": {"train_csv": paths["train"], "val_csv": paths["val"], "test_csv": paths["test"]},
        "mlflow": {"experiment": "exp-baseline"},
        "baseline": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 4],
            "min_df": 1,
            "max_features": 500,
            "classifier": "logreg",
            "C": 1.0,
            "max_iter": 500,
            "class_weight_balanced": True,
        },
    }
    cfg_path = tracking / "training.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = baseline.run(str(cfg_path), classifier=classifier)

    assert 0.0 <= metrics["f1_macro"] <= 1.0
    runs = mlflow.search_runs(experiment_names=["exp-baseline"])
    assert len(runs) == 1
    assert runs["params.classifier"].iloc[0] == classifier
    assert "metrics.test_f1_macro" in runs.columns
