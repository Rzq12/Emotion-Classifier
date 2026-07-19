"""Embed processed reviews into the ChromaDB vector store.

Offline step (Fase 3): runs once after the dataset is prepared; the API later
loads the persisted store read-only.

Run:
    python -m src.rag.build_index --config configs/rag.yaml [--reset]
"""

from __future__ import annotations

import argparse

import yaml

from src.rag.corpus import load_reviews
from src.rag.embeddings import Embedder
from src.rag.vector_store import ReviewVectorStore


def build(config_path: str, reset: bool = False) -> int:
    """Embed all reviews and upsert them into ChromaDB. Returns review count.

    With ``reset=True`` the collection is dropped first so vectors from a
    previous dataset version cannot linger (upsert alone never deletes).
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reviews = load_reviews(cfg["data"]["csvs"])
    embedder = Embedder(
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    store = ReviewVectorStore(
        persist_dir=cfg["vector_store"]["persist_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
    )
    if reset:
        store.reset()

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
    parser.add_argument(
        "--reset", action="store_true", help="Drop the collection before indexing."
    )
    args = parser.parse_args()
    build(args.config, reset=args.reset)


if __name__ == "__main__":
    main()
