"""Build the processed emotion dataset from raw review CSVs.

Pipeline (Fase 1):
    raw train/test CSV
        -> clean text (src.data.preprocessing)
        -> normalize labels to canonical 3-class schema (src.data.labels)
        -> drop empties / duplicates
        -> stratified train/val split (test kept as provided)
        -> write data/processed/{train,val,test}.csv

Run:
    python -m src.data.prepare_dataset --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.labels import normalize_label
from src.data.preprocessing import PreprocessConfig, clean_text

# Canonical columns of the processed dataset.
TEXT_COL = "text"
LABEL_COL = "label"


def load_config(path: str | Path) -> dict:
    """Load the YAML data config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_preprocess_config(cfg: dict) -> PreprocessConfig:
    p = cfg["preprocessing"]
    return PreprocessConfig(
        lowercase=p["lowercase"],
        remove_urls=p["remove_urls"],
        remove_mentions=p["remove_mentions"],
        reduce_repeated_chars=p["reduce_repeated_chars"],
        normalize_slang=p["normalize_slang"],
        emoji_strategy=p["emoji_strategy"],
        min_chars=p["min_chars"],
    )


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    pre_cfg: PreprocessConfig,
) -> pd.DataFrame:
    """Clean text, normalize labels, and drop empty/short/duplicate rows.

    Returns a new frame with canonical columns ``[text, label]``.
    """
    out = pd.DataFrame(
        {
            TEXT_COL: df[text_column].map(lambda t: clean_text(t, pre_cfg)),
            LABEL_COL: df[label_column].map(normalize_label),
        }
    )

    # Remove rows that became empty/too short after cleaning, then dedupe.
    out = out[out[TEXT_COL].str.len() >= pre_cfg.min_chars]
    out = out.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    return out


def split_train_val(
    df: pd.DataFrame,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/val split on the label column."""
    stratify_col = df[LABEL_COL] if stratify else None
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _label_summary(df: pd.DataFrame) -> str:
    counts = df[LABEL_COL].value_counts().to_dict()
    return f"n={len(df)} {counts}"


def prepare(config_path: str | Path) -> dict[str, pd.DataFrame]:
    """Execute the full preparation pipeline and write processed CSVs."""
    cfg = load_config(config_path)
    pre_cfg = _build_preprocess_config(cfg)

    raw = cfg["raw"]
    text_column = raw["text_column"]
    label_column = raw["label_column"]

    train_raw = pd.read_csv(raw["train_csv"])
    test_raw = pd.read_csv(raw["test_csv"])

    train_clean = clean_dataframe(train_raw, text_column, label_column, pre_cfg)
    test_clean = clean_dataframe(test_raw, text_column, label_column, pre_cfg)

    split_cfg = cfg["split"]
    train_df, val_df = split_train_val(
        train_clean,
        val_size=split_cfg["val_size"],
        random_state=split_cfg["random_state"],
        stratify=split_cfg["stratify"],
    )

    out_dir = Path(cfg["processed"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(cfg["processed"]["train_csv"], index=False)
    val_df.to_csv(cfg["processed"]["val_csv"], index=False)
    test_clean.to_csv(cfg["processed"]["test_csv"], index=False)

    print("[prepare_dataset] processed dataset written:")
    print(f"  train -> {cfg['processed']['train_csv']} | {_label_summary(train_df)}")
    print(f"  val   -> {cfg['processed']['val_csv']} | {_label_summary(val_df)}")
    print(f"  test  -> {cfg['processed']['test_csv']} | {_label_summary(test_clean)}")

    return {"train": train_df, "val": val_df, "test": test_clean}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed emotion dataset.")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml).",
    )
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
