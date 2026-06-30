"""Fine-tune IndoBERT for 3-class emotion classification.

Handles the dataset imbalance with class-weighted cross-entropy and logs
params, metrics, and a confusion matrix to MLflow. The trained model + tokenizer
are saved to ``output_dir`` for serving (and optional HF Hub push later).

Run (full):
    python -m src.training.train_indobert --config configs/training.yaml

Run (smoke test, tiny subset):
    python -m src.training.train_indobert --config configs/training.yaml \
        --max-train-samples 64 --epochs 1
"""

from __future__ import annotations

import argparse

import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.labels import ID2LABEL, LABEL2ID, LABELS
from src.tracking.mlflow_utils import log_confusion_matrix, setup_tracking
from src.training.dataset import Split, load_dataset
from src.training.metrics import compute_metrics, confusion_matrix_df


class _TextDataset(torch.utils.data.Dataset):
    """Wrap pre-tokenized encodings + labels for the HF Trainer."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Trainer using class-weighted cross-entropy to counter label imbalance."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights normalized to mean 1.0."""
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero
    weights = counts.sum() / (len(LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float)


def _hf_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics(labels, preds)


def _maybe_subsample(split: Split, n: int | None) -> Split:
    if n is None or n >= len(split):
        return split
    return Split(texts=split.texts[:n], labels=split.labels[:n])


def run(config_path: str, overrides: dict | None = None) -> dict[str, float]:
    """Fine-tune IndoBERT and log everything to MLflow."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ic = cfg["indobert"]
    if overrides:
        ic.update({k: v for k, v in overrides.items() if v is not None})

    set_seed(ic["seed"])

    data = load_dataset(cfg["data"]["train_csv"], cfg["data"]["val_csv"], cfg["data"]["test_csv"])
    train_split = _maybe_subsample(data.train, ic.get("max_train_samples"))

    tokenizer = AutoTokenizer.from_pretrained(ic["model_name"])

    def tok(texts: list[str]) -> dict:
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=ic["max_length"],
        )

    train_ds = _TextDataset(tok(train_split.texts), train_split.labels)
    val_ds = _TextDataset(tok(data.val.texts), data.val.labels)
    test_ds = _TextDataset(tok(data.test.texts), data.test.labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        ic["model_name"],
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class_weights = _compute_class_weights(train_split.labels) if ic["class_weighting"] else None

    args = TrainingArguments(
        output_dir=ic["output_dir"],
        num_train_epochs=ic["num_epochs"],
        per_device_train_batch_size=ic["batch_size"],
        per_device_eval_batch_size=ic["eval_batch_size"],
        learning_rate=float(ic["learning_rate"]),
        weight_decay=ic["weight_decay"],
        warmup_ratio=ic["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=ic["seed"],
    )

    setup_tracking(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=ic["run_name"]):
        mlflow.set_tag("model_type", "indobert")
        mlflow.log_params(
            {
                "model_name": ic["model_name"],
                "max_length": ic["max_length"],
                "num_epochs": ic["num_epochs"],
                "batch_size": ic["batch_size"],
                "learning_rate": ic["learning_rate"],
                "weight_decay": ic["weight_decay"],
                "warmup_ratio": ic["warmup_ratio"],
                "class_weighting": ic["class_weighting"],
                "train_samples": len(train_split),
            }
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_hf_metrics,
            class_weights=class_weights,
        )
        trainer.train()

        val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
        mlflow.log_metrics({k: v for k, v in val_metrics.items() if isinstance(v, (int, float))})

        test_pred = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
        test_metrics = compute_metrics(data.test.labels, test_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log_confusion_matrix(
            confusion_matrix_df(data.test.labels, test_pred),
            artifact_name="test_confusion_matrix",
        )

        trainer.save_model(ic["output_dir"])
        tokenizer.save_pretrained(ic["output_dir"])
        mlflow.log_artifacts(ic["output_dir"], artifact_path="model")

        print(f"[indobert] test F1-macro={test_metrics['f1_macro']:.4f} acc={test_metrics['accuracy']:.4f}")

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT for emotion classification.")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run(
        args.config,
        overrides={"max_train_samples": args.max_train_samples, "num_epochs": args.epochs},
    )


if __name__ == "__main__":
    main()
