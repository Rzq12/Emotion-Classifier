"""Compare MLflow runs and promote the best model to the registry.

Selection metric: ``test_f1_macro`` (primary metric for this imbalanced task).
Requires a database tracking backend (SQLite by default) for the model registry.

Run:
    python -m src.tracking.registry --experiment indo-review-emotion
    python -m src.tracking.registry --register   # also register the best run
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd

from src.tracking.mlflow_utils import DEFAULT_EXPERIMENT, setup_tracking

REGISTERED_MODEL_NAME = "indo-emotion-classifier"
SELECTION_METRIC = "test_f1_macro"


def list_runs(experiment: str = DEFAULT_EXPERIMENT) -> pd.DataFrame:
    """Return a comparison table of runs sorted by the selection metric (desc)."""
    setup_tracking(experiment)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"Experiment '{experiment}' not found.")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        return runs

    cols = ["run_id", "tags.mlflow.runName", "tags.model_type"]
    metric_cols = [
        f"metrics.{m}"
        for m in ("test_f1_macro", "test_accuracy", "test_f1_anger", "test_f1_sadness", "test_f1_happiness")
        if f"metrics.{m}" in runs.columns
    ]
    table = runs[[c for c in cols if c in runs.columns] + metric_cols]
    sort_key = f"metrics.{SELECTION_METRIC}"
    if sort_key in table.columns:
        table = table.sort_values(sort_key, ascending=False)
    return table.reset_index(drop=True)


def register_best(experiment: str = DEFAULT_EXPERIMENT) -> str | None:
    """Register the best run's model artifact and tag it as production candidate."""
    setup_tracking(experiment)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"Experiment '{experiment}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
        max_results=1,
    )
    if runs.empty:
        print("[registry] no runs found.")
        return None

    best = runs.iloc[0]
    run_id = best["run_id"]
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, result.version, "stage", "production-candidate"
    )
    score = best.get(f"metrics.{SELECTION_METRIC}")
    print(
        f"[registry] registered '{REGISTERED_MODEL_NAME}' v{result.version} "
        f"from run {run_id} ({SELECTION_METRIC}={score})"
    )
    return result.version


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs / register best model.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--register", action="store_true", help="Register the best run's model.")
    args = parser.parse_args()

    table = list_runs(args.experiment)
    if table.empty:
        print("[registry] no runs to compare.")
        return
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))

    if args.register:
        register_best(args.experiment)


if __name__ == "__main__":
    main()
