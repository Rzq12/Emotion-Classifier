"""Shared corpus loading for all retrieval indexes (vector + BM25).

Review IDs are content-hashed (``{split}_{sha1(text)[:10]}``) so they stay
stable when the dataset is regenerated with shifted row order — positional IDs
would silently point at different reviews after a rebuild.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def review_id(split: str, text: str) -> str:
    """Stable content-based ID for one review."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{split}_{digest}"


def load_reviews(csv_paths: list[str]) -> pd.DataFrame:
    """Concatenate processed CSVs with ``split`` and content-hash ``review_id``."""
    frames = []
    for path in csv_paths:
        split = Path(path).stem  # train / val / test
        df = pd.read_csv(path)
        df["text"] = df["text"].astype(str)
        df["split"] = split
        df["review_id"] = [review_id(split, t) for t in df["text"]]
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    # Content-hash IDs collide only for identical text within a split; keep the
    # first occurrence so index upserts stay one-vector-per-ID.
    return merged.drop_duplicates(subset="review_id", keep="first").reset_index(drop=True)
