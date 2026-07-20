"""Baseline: frozen MiniLM sentence embeddings + Logistic Regression.

Reuses the same multilingual MiniLM the RAG layer uses (src.rag.embeddings), so
no new model download. Trains in seconds on CPU — the cheapest "semantic"
middle ground between TF-IDF and full transformer fine-tuning.

Run:
    python -m src.training.embedding_baseline --config configs/training.yaml
"""

from __future__ import annotations

import argparse

import mlflow
import yaml
from sklearn.linear_model import LogisticRegression

from src.rag.embeddings import Embedder
from src.tracking.mlflow_utils import log_confusion_matrix, setup_tracking
from src.training.dataset import load_dataset
from src.training.metrics import compute_metrics, confusion_matrix_df


def run(config_path: str) -> dict[str, float]:
    """Encode splits with MiniLM, fit LogReg, log everything to MLflow."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b = cfg["embed_baseline"]

    data = load_dataset(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    )
    setup_tracking(cfg["mlflow"]["experiment"])

    embedder = Embedder(model_name=b["model_name"], batch_size=b["batch_size"])
    clf = LogisticRegression(
        C=b["C"],
        max_iter=b["max_iter"],
        class_weight="balanced" if b["class_weight_balanced"] else None,
    )

    with mlflow.start_run(run_name="minilm-embed-logreg"):
        mlflow.set_tag("model_type", "embed_baseline")
        mlflow.log_params({"classifier": "logistic_regression", **b})

        x_train = embedder.encode(data.train.texts)
        x_val = embedder.encode(data.val.texts)
        x_test = embedder.encode(data.test.texts)

        clf.fit(x_train, data.train.labels)
        val_metrics = compute_metrics(data.val.labels, clf.predict(x_val))
        test_pred = clf.predict(x_test)
        test_metrics = compute_metrics(data.test.labels, test_pred)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log_confusion_matrix(
            confusion_matrix_df(data.test.labels, test_pred),
            artifact_name="test_confusion_matrix",
        )

        print(
            f"[minilm+logreg] val F1-macro={val_metrics['f1_macro']:.4f} "
            f"test F1-macro={test_metrics['f1_macro']:.4f} acc={test_metrics['accuracy']:.4f}"
        )

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLM embeddings + LogReg baseline.")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
