"""Baseline emotion classifier: TF-IDF + Logistic Regression.

This is the mandatory classical baseline (CLAUDE.md): IndoBERT must beat it on
F1-macro before we claim it adds value. Fast, CPU-friendly, fully logged to MLflow.

Run:
    python -m src.training.baseline --config configs/training.yaml
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.tracking.mlflow_utils import log_confusion_matrix, setup_tracking
from src.training.dataset import load_dataset
from src.training.metrics import compute_metrics, confusion_matrix_df


def build_pipeline(cfg: dict) -> Pipeline:
    """Construct the TF-IDF + LogisticRegression pipeline from config."""
    b = cfg["baseline"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(b["ngram_range"]),
                    min_df=b["min_df"],
                    max_features=b["max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=b["C"],
                    max_iter=b["max_iter"],
                    class_weight="balanced" if b["class_weight_balanced"] else None,
                ),
            ),
        ]
    )


def run(config_path: str) -> dict[str, float]:
    """Train the baseline, evaluate on val + test, and log everything to MLflow."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_dataset(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    )

    setup_tracking(cfg["mlflow"]["experiment"])
    pipeline = build_pipeline(cfg)

    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_params(
            {
                "vectorizer": "tfidf",
                "classifier": "logistic_regression",
                **{f"tfidf_{k}": v for k, v in cfg["baseline"].items()},
            }
        )

        pipeline.fit(data.train.texts, data.train.labels)

        val_pred = pipeline.predict(data.val.texts)
        test_pred = pipeline.predict(data.test.texts)

        val_metrics = compute_metrics(data.val.labels, val_pred)
        test_metrics = compute_metrics(data.test.labels, test_pred)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        log_confusion_matrix(
            confusion_matrix_df(data.test.labels, test_pred),
            artifact_name="test_confusion_matrix",
        )

        # Persist the fitted pipeline as an artifact for later comparison.
        model_path = "baseline_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        print(f"[baseline] val  F1-macro={val_metrics['f1_macro']:.4f} acc={val_metrics['accuracy']:.4f}")
        print(f"[baseline] test F1-macro={test_metrics['f1_macro']:.4f} acc={test_metrics['accuracy']:.4f}")
        per_class = ", ".join(
            f"{k.replace('f1_', '')}={v:.3f}"
            for k, v in test_metrics.items()
            if k.startswith("f1_") and k not in ("f1_macro", "f1_weighted")
        )
        print(f"[baseline] per-class test F1: {per_class}")

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogReg baseline.")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
