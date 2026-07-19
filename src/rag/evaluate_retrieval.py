"""Retrieval evaluation against a small golden set (vector-only vs hybrid).

Metrics per mode: hit@1, hit@5, MRR@k — "relevant" means the retrieved text
contains one of the case's expected terms. Also reports top-1 cosine scores for
in-domain vs out-of-domain probes to ground the chat ``min_score`` threshold.

Run:
    python -m src.rag.evaluate_retrieval --config configs/rag.yaml \
        --eval-config configs/retrieval_eval.yaml
"""

from __future__ import annotations

import argparse

import yaml

from src.rag.bm25 import BM25Index
from src.rag.corpus import load_reviews
from src.rag.embeddings import Embedder
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import RetrievedReview, ReviewVectorStore


def _is_relevant(review: RetrievedReview, expected_terms: list[str]) -> bool:
    text = review.text.lower()
    return any(term.lower() in text for term in expected_terms)


def evaluate(retriever: HybridRetriever, cases: list[dict], top_k: int) -> dict:
    """Return aggregate metrics over the golden set for one retriever."""
    hits1 = hits5 = 0
    reciprocal_ranks: list[float] = []

    for case in cases:
        results = retriever.query(case["query"], n_results=top_k)
        first_rank = next(
            (i + 1 for i, r in enumerate(results) if _is_relevant(r, case["expected_terms"])),
            None,
        )
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
            hits1 += first_rank == 1
            hits5 += first_rank <= 5
        else:
            reciprocal_ranks.append(0.0)

    n = len(cases)
    return {
        "hit@1": round(hits1 / n, 3),
        "hit@5": round(hits5 / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
    }


def top1_cosine(retriever: HybridRetriever, query: str) -> float:
    results = retriever.query(query, n_results=1)
    return results[0].score if results else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate review retrieval quality.")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--eval-config", default="configs/retrieval_eval.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.eval_config, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    store = ReviewVectorStore(
        persist_dir=cfg["vector_store"]["persist_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
    )
    embedder = Embedder(
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
    )
    reviews = load_reviews(cfg["data"]["csvs"])
    bm25 = BM25Index()
    bm25.fit(
        ids=reviews["review_id"].tolist(),
        texts=reviews["text"].tolist(),
        emotions=reviews["label"].tolist(),
    )

    vector_only = HybridRetriever(store, embedder, bm25=None)
    hybrid = HybridRetriever(store, embedder, bm25=bm25)

    cases = eval_cfg["cases"]
    top_k = eval_cfg.get("top_k", 10)

    print(f"Golden set: {len(cases)} kasus | top_k={top_k} | korpus={len(reviews)}\n")
    print(f"{'Mode':<14}{'hit@1':>8}{'hit@5':>8}{'MRR':>8}")
    for name, retriever in (("vector-only", vector_only), ("hybrid", hybrid)):
        m = evaluate(retriever, cases, top_k)
        print(f"{name:<14}{m['hit@1']:>8}{m['hit@5']:>8}{m['mrr']:>8}")

    print("\nKalibrasi min_score (cosine top-1):")
    in_domain = [top1_cosine(vector_only, c["query"]) for c in cases]
    print(f"  in-domain : min={min(in_domain):.3f} mean={sum(in_domain)/len(in_domain):.3f}")
    probes = eval_cfg.get("irrelevant_probes", [])
    if probes:
        out_domain = [top1_cosine(vector_only, q) for q in probes]
        for q, s in zip(probes, out_domain, strict=True):
            print(f"  out-domain: {s:.3f}  ({q})")
        print(
            f"  -> pilih min_score di antara max out-domain ({max(out_domain):.3f}) "
            f"dan min in-domain ({min(in_domain):.3f})"
        )


if __name__ == "__main__":
    main()
