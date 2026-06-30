"""Embed processed reviews into the ChromaDB vector store.

Offline step (Fase 3): runs once after the dataset is prepared; the API later
loads the persisted store read-only.

Run:
    python -m src.rag.build_index --config configs/rag.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.rag.embeddings import Embedder
from src.rag.vector_store import ReviewVectorStore


def _load_reviews(csv_paths: list[str]) -> pd.DataFrame:
    """Concatenate processed CSVs and assign a stable review_id per split."""
    frames = []
    for path in csv_paths:
        split = Path(path).stem  # train / val / test
        df = pd.read_csv(path)
        df["split"] = split
        df["review_id"] = [f"{split}_{i}" for i in range(len(df))]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build(config_path: str) -> int:
    """Embed all reviews and upsert them into ChromaDB. Returns review count."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reviews = _load_reviews(cfg["data"]["csvs"])
    embedder = Embedder(
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    store = ReviewVectorStore(
        persist_dir=cfg["vector_store"]["persist_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
    )

    texts = reviews["text"].astype(str).tolist()
    embeddings = embedder.encode(texts)
    metadatas = [
        {"emotion": row["label"], "split": row["split"]}
        for _, row in reviews.iterrows()
    ]

    store.add(
        ids=reviews["review_id"].tolist(),
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    total = store.count()
    print(f"[build_index] indexed {len(reviews)} reviews; collection now has {total}.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB review index.")
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
