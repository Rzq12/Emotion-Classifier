"""SetFit few-shot baseline on multilingual MiniLM.

Contrastive fine-tuning of the sentence-transformer body + LogReg head using a
small per-class sample (classic SetFit setting). CPU-friendly, and an
interesting portfolio comparison: how close does few-shot get to full-data
baselines?

Run:
    python -m src.training.setfit_baseline --config configs/training.yaml
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd
import yaml

from src.tracking.mlflow_utils import log_confusion_matrix, setup_tracking
from src.training.dataset import load_dataset
from src.training.metrics import compute_metrics, confusion_matrix_df


def _sample_per_class(texts: list[str], labels: list[str], n: int, seed: int):
    """Take up to ``n`` examples per class (deterministic)."""
    df = pd.DataFrame({"text": texts, "label": labels})
    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(n, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    return sampled["text"].tolist(), sampled["label"].tolist()


def run(config_path: str) -> dict[str, float]:
    """Train SetFit on a per-class sample and log evaluation to MLflow."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s = cfg["setfit"]

    data = load_dataset(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    )
    setup_tracking(cfg["mlflow"]["experiment"])

    train_texts, train_labels = _sample_per_class(
        data.train.texts, data.train.labels, n=s["samples_per_class"], seed=s["seed"]
    )
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

    model = SetFitModel.from_pretrained(s["model_name"])
    args = TrainingArguments(
        num_epochs=s["num_epochs"],
        batch_size=s["batch_size"],
        seed=s["seed"],
    )

    with mlflow.start_run(run_name="setfit-minilm-fewshot"):
        mlflow.set_tag("model_type", "setfit")
        mlflow.log_params({**s, "n_train_sampled": len(train_texts)})

        trainer = Trainer(model=model, args=args, train_dataset=train_ds)
        trainer.train()

        val_metrics = compute_metrics(data.val.labels, list(model.predict(data.val.texts)))
        test_pred = list(model.predict(data.test.texts))
        test_metrics = compute_metrics(data.test.labels, test_pred)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log_confusion_matrix(
            confusion_matrix_df(data.test.labels, test_pred),
            artifact_name="test_confusion_matrix",
        )

        print(
            f"[setfit] ({len(train_texts)} sampel) val F1-macro={val_metrics['f1_macro']:.4f} "
            f"test F1-macro={test_metrics['f1_macro']:.4f} acc={test_metrics['accuracy']:.4f}"
        )

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SetFit few-shot baseline (MiniLM).")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
