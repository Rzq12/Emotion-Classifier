"""Baseline emotion classifier: TF-IDF (word + char n-gram) + classic classifier.

This is the mandatory classical baseline (CLAUDE.md): IndoBERT must beat it on
F1-macro before we claim it adds value. Fast, CPU-friendly, fully logged to MLflow.
Char n-grams (char_wb) are robust to the typos/slang common in app reviews
("gabisa", "gbs"); the classifier is selectable (logreg | linearsvc).

Run:
    python -m src.training.baseline --config configs/training.yaml
    python -m src.training.baseline --config configs/training.yaml --classifier linearsvc
"""

from __future__ import annotations

import argparse

import joblib
import mlflow
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.tracking.mlflow_utils import log_confusion_matrix, setup_tracking
from src.training.dataset import load_dataset
from src.training.metrics import compute_metrics, confusion_matrix_df


def _build_vectorizer(b: dict) -> TfidfVectorizer | FeatureUnion:
    """Word TF-IDF, optionally unioned with a char_wb TF-IDF."""
    word = TfidfVectorizer(
        ngram_range=tuple(b["word_ngram_range"]),
        min_df=b["min_df"],
        max_features=b["max_features"],
        sublinear_tf=True,
    )
    if not b.get("char_ngram_range"):
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(b["char_ngram_range"]),
        min_df=b["min_df"],
        max_features=b["max_features"],
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])


def _build_classifier(b: dict):
    """Instantiate the configured classifier (logreg | linearsvc)."""
    class_weight = "balanced" if b["class_weight_balanced"] else None
    name = b.get("classifier", "logreg")
    if name == "logreg":
        return LogisticRegression(C=b["C"], max_iter=b["max_iter"], class_weight=class_weight)
    if name == "linearsvc":
        return LinearSVC(C=b["C"], max_iter=b["max_iter"], class_weight=class_weight)
    raise ValueError(f"Unknown baseline classifier '{name}' (expected logreg | linearsvc).")


def build_pipeline(cfg: dict) -> Pipeline:
    """Construct the TF-IDF + classifier pipeline from config."""
    b = cfg["baseline"]
    return Pipeline([("tfidf", _build_vectorizer(b)), ("clf", _build_classifier(b))])


def run(config_path: str, classifier: str | None = None) -> dict[str, float]:
    """Train the baseline, evaluate on val + test, and log everything to MLflow."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if classifier:
        cfg["baseline"]["classifier"] = classifier

    data = load_dataset(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    )

    setup_tracking(cfg["mlflow"]["experiment"])
    pipeline = build_pipeline(cfg)

    b = cfg["baseline"]
    features = "tfidf-word-char" if b.get("char_ngram_range") else "tfidf-word"
    run_name = f"baseline-{features}-{b.get('classifier', 'logreg')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_params(
            {
                "vectorizer": features,
                "classifier": b.get("classifier", "logreg"),
                **{f"tfidf_{k}": v for k, v in b.items()},
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
    parser = argparse.ArgumentParser(description="Train classical TF-IDF baseline.")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument(
        "--classifier",
        choices=["logreg", "linearsvc"],
        default=None,
        help="Override the classifier from config.",
    )
    args = parser.parse_args()
    run(args.config, classifier=args.classifier)


if __name__ == "__main__":
    main()
