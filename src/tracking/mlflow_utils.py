"""Thin helpers around MLflow so training scripts stay clean.

Keeps tracking-URI/experiment setup and confusion-matrix artifact logging in one
place. Tracking URI defaults to the local ``./mlruns`` dir (overridable via the
``MLFLOW_TRACKING_URI`` env var), matching ``.env.example``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "indo-review-emotion"


def setup_tracking(experiment: str = DEFAULT_EXPERIMENT) -> None:
    """Configure tracking URI (from env or default) and active experiment.

    Loads ``.env`` first so remote tracking (e.g. DagsHub via
    ``MLFLOW_TRACKING_URI`` + ``MLFLOW_TRACKING_USERNAME``/``PASSWORD``) works
    from training scripts too — previously only the API loaded dotenv, so
    training silently fell back to the local SQLite store.
    """
    load_dotenv()
    # MLflow 3.x prints an emoji banner on run end; Windows cp1252 consoles
    # raise UnicodeEncodeError there, killing the process before the run is
    # marked FINISHED. Force UTF-8 (best effort) so logging always completes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def log_confusion_matrix(cm_df: pd.DataFrame, artifact_name: str = "confusion_matrix") -> None:
    """Log a confusion matrix as both a CSV and a PNG heatmap artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        csv_path = tmp_path / f"{artifact_name}.csv"
        cm_df.to_csv(csv_path)
        mlflow.log_artifact(str(csv_path))

        png_path = tmp_path / f"{artifact_name}.png"
        _save_heatmap(cm_df, png_path, title=artifact_name)
        mlflow.log_artifact(str(png_path))


def _save_heatmap(cm_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Render a simple confusion-matrix heatmap without seaborn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_df.values, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)
    ax.set_title(title)

    # Annotate each cell with its count.
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = cm_df.values[i, j]
            color = "white" if value > cm_df.values.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
