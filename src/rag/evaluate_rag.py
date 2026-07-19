"""End-to-end RAG evaluation with the REAL index and REAL LLM (no fakes).

Layers evaluated:
1. Retrieval — hit@1/hit@5/MRR on the golden set (vector-only vs hybrid).
2. Chat E2E — real ChatResponder + LLM per golden query: answer produced,
   citation validity, groundedness scored 1-5 by an LLM judge, latency.
3. Refusal — out-of-domain probes must be declined (gate or LLM).
4. Insight E2E — real InsightGenerator per focus query: schema validity,
   example-ID validity, groundedness of the summary, latency, cache hit.

Requires an LLM API key (Groq/Gemini) in .env. Writes a markdown report.

Run:
    python -m src.rag.evaluate_rag [--output reports/rag_eval_report.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.llm.base import LLMClient, LLMError
from src.llm.factory import get_llm_client
from src.llm.prompt_loader import render_prompt
from src.rag.bm25 import BM25Index
from src.rag.chat import NO_CONTEXT_MESSAGE, ChatConfig, ChatResponder
from src.rag.corpus import load_reviews
from src.rag.embeddings import Embedder
from src.rag.evaluate_retrieval import evaluate as evaluate_retrieval_metrics
from src.rag.formatting import format_reviews
from src.rag.insight import InsightConfig, InsightGenerator
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import ReviewVectorStore

INSIGHT_QUERIES = [
    "keluhan utama pengguna",
    "masalah pembayaran dan transaksi",
    "keluhan seputar layanan dokter dan konsultasi",
]


def judge_groundedness(llm: LLMClient, reviews_block: str, question: str, answer: str) -> dict:
    """Score answer groundedness 1-5 via the LLM judge prompt."""
    prompt = render_prompt(
        "judge.txt", reviews=reviews_block, question=question, answer=answer
    )
    raw = llm.generate(prompt, temperature=0.0, max_tokens=200)
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start : end + 1])
        return {"score": int(data.get("score", 0)), "alasan": str(data.get("alasan", ""))}
    except (json.JSONDecodeError, ValueError):
        return {"score": 0, "alasan": f"judge output tidak terparse: {raw[:80]}"}


def evaluate_chat(
    responder: ChatResponder,
    retriever: HybridRetriever,
    llm: LLMClient,
    cases: list[dict],
) -> list[dict]:
    """Run every golden query through the real chat pipeline and judge it."""
    rows = []
    for case in cases:
        question = case["query"]
        retrieved = retriever.query(question, n_results=responder.config.top_k)
        t0 = time.perf_counter()
        out = responder.answer(question)
        latency = time.perf_counter() - t0

        answered = out["answer"] != NO_CONTEXT_MESSAGE and bool(out["answer"].strip())
        retrieved_ids = {r.review_id for r in retrieved}
        sources_valid = set(out["sources"]) <= retrieved_ids if out["sources"] else answered

        judge = judge_groundedness(
            llm, format_reviews(retrieved), question, out["answer"]
        )
        rows.append(
            {
                "query": question,
                "answered": answered,
                "n_sources": len(out["sources"]),
                "sources_valid": bool(sources_valid),
                "groundedness": judge["score"],
                "alasan": judge["alasan"],
                "latency_s": round(latency, 2),
            }
        )
    return rows


def evaluate_refusal(responder: ChatResponder, probes: list[str]) -> list[dict]:
    """Out-of-domain probes should be refused by the gate or by the LLM."""
    rows = []
    for probe in probes:
        out = responder.answer(probe)
        refused = NO_CONTEXT_MESSAGE.rstrip(".") in out["answer"]
        rows.append(
            {
                "query": probe,
                "refused": refused,
                "n_sources": len(out["sources"]),
                "answer_head": out["answer"][:90],
            }
        )
    return rows


def evaluate_insight(
    generator: InsightGenerator,
    retriever: HybridRetriever,
    llm: LLMClient,
    queries: list[str],
) -> list[dict]:
    """Run real insight generation and validate structure + grounding."""
    required = {"summary", "themes", "sample_quotes", "recommendations", "note", "n_reviews"}
    rows = []
    for query in queries:
        retrieved = retriever.query(
            query,
            n_results=generator.config.top_k,
            emotions=generator.config.negative_emotions,
        )
        retrieved_ids = {r.review_id for r in retrieved}

        # First call populates the cache (cold, real LLM); second must hit it.
        t0 = time.perf_counter()
        result = generator.generate(query, use_cache=True)
        latency = time.perf_counter() - t0

        example_ids = [
            i
            for theme in result.get("themes", [])
            if isinstance(theme, dict)
            for i in theme.get("example_review_ids", [])
        ]
        judge = judge_groundedness(
            llm, format_reviews(retrieved), f"Insight: {query}", result.get("summary", "")
        )
        cached_run = generator.generate(query, use_cache=True)

        rows.append(
            {
                "query": query,
                "schema_valid": required <= set(result),
                "n_themes": len(result.get("themes", [])),
                "example_ids_valid": all(i in retrieved_ids for i in example_ids),
                "n_example_ids": len(example_ids),
                "groundedness": judge["score"],
                "latency_s": round(latency, 2),
                "cache_works": bool(cached_run.get("cached")),
            }
        )
    return rows


def render_report(
    retrieval: dict,
    chat_rows: list[dict],
    refusal_rows: list[dict],
    insight_rows: list[dict],
    meta: dict,
) -> str:
    """Render all evaluation results as one markdown report."""
    lines = [
        "# RAG Evaluation Report — Indo Review Intelligence",
        "",
        f"Generated: {meta['generated_at']} | LLM: {meta['llm']} | korpus: {meta['corpus']} review",
        "",
        "Evaluasi end-to-end dengan index ChromaDB asli dan panggilan LLM sungguhan",
        "(bukan mock). Groundedness dinilai LLM-as-judge (skala 1-5).",
        "",
        "## 1. Retrieval (golden set)",
        "",
        "| Mode | hit@1 | hit@5 | MRR |",
        "|---|---|---|---|",
    ]
    for mode, m in retrieval.items():
        lines.append(f"| {mode} | {m['hit@1']} | {m['hit@5']} | {m['mrr']} |")

    grounded = [r["groundedness"] for r in chat_rows if r["groundedness"] > 0]
    lines += [
        "",
        "## 2. Chat end-to-end",
        "",
        f"- Terjawab: **{sum(r['answered'] for r in chat_rows)}/{len(chat_rows)}**",
        f"- Sources valid (⊆ retrieved): **{sum(r['sources_valid'] for r in chat_rows)}"
        f"/{len(chat_rows)}**",
        f"- Groundedness rata-rata: **{statistics.mean(grounded):.2f}/5** "
        f"(min {min(grounded)}, max {max(grounded)})"
        if grounded
        else "- Groundedness: judge gagal",
        f"- Latency rata-rata: **{statistics.mean(r['latency_s'] for r in chat_rows):.2f}s**",
        "",
        "| Query | Jawab | Sumber | Grounded | Latency |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['query']} | {'ya' if r['answered'] else 'tolak'} | {r['n_sources']} "
        f"| {r['groundedness']}/5 | {r['latency_s']}s |"
        for r in chat_rows
    ]

    lines += [
        "",
        "## 3. Penolakan query di luar domain",
        "",
        f"Ditolak dengan benar: **{sum(r['refused'] for r in refusal_rows)}/{len(refusal_rows)}**",
        "",
        "| Probe | Ditolak | Sources |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['query']} | {'ya' if r['refused'] else 'TIDAK'} | {r['n_sources']} |"
        for r in refusal_rows
    ]

    ins_grounded = [r["groundedness"] for r in insight_rows if r["groundedness"] > 0]
    lines += [
        "",
        "## 4. Insight end-to-end",
        "",
        f"- Schema valid: **{sum(r['schema_valid'] for r in insight_rows)}/{len(insight_rows)}**",
        f"- Semua example_review_ids valid: "
        f"**{sum(r['example_ids_valid'] for r in insight_rows)}/{len(insight_rows)}**",
        f"- Cache bekerja: **{sum(r['cache_works'] for r in insight_rows)}/{len(insight_rows)}**",
        f"- Groundedness summary rata-rata: **{statistics.mean(ins_grounded):.2f}/5**"
        if ins_grounded
        else "- Groundedness: judge gagal",
        "",
        "| Fokus | Tema | Contoh ID (valid) | Grounded | Latency |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['query']} | {r['n_themes']} | {r['n_example_ids']} "
        f"({'ya' if r['example_ids_valid'] else 'TIDAK'}) | {r['groundedness']}/5 "
        f"| {r['latency_s']}s |"
        for r in insight_rows
    ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation (real LLM).")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--eval-config", default="configs/retrieval_eval.yaml")
    parser.add_argument("--provider", default=None, help="Override LLM_PROVIDER.")
    parser.add_argument("--output", default="reports/rag_eval_report.md")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.eval_config, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    import os

    provider = args.provider or os.getenv("LLM_PROVIDER", "groq")
    llm = get_llm_client(provider)
    if not llm.is_available():
        raise SystemExit(f"LLM provider '{provider}' belum terkonfigurasi (API key kosong).")

    store = ReviewVectorStore(
        persist_dir=cfg["vector_store"]["persist_dir"],
        collection_name=cfg["vector_store"]["collection_name"],
    )
    embedder = Embedder(cfg["embedding"]["model_name"], cfg["embedding"]["batch_size"])
    reviews = load_reviews(cfg["data"]["csvs"])
    bm25 = BM25Index()
    bm25.fit(
        ids=reviews["review_id"].tolist(),
        texts=reviews["text"].tolist(),
        emotions=reviews["label"].tolist(),
    )
    retriever = HybridRetriever(store, embedder, bm25)
    vector_only = HybridRetriever(store, embedder, bm25=None)

    cc = cfg["chat"]
    responder = ChatResponder(
        retriever,
        llm,
        ChatConfig(
            top_k=cc["top_k"],
            min_score=cc.get("min_score", 0.55),
            min_bm25=cc.get("min_bm25", 1.0),
        ),
    )
    ic = cfg["insight"]
    generator = InsightGenerator(
        retriever,
        llm,
        InsightConfig(
            negative_emotions=ic["negative_emotions"],
            top_k=ic["top_k"],
            cache_ttl_seconds=ic["cache_ttl_seconds"],
            cache_max_size=ic["cache_max_size"],
        ),
    )

    cases = eval_cfg["cases"]
    probes = eval_cfg.get("irrelevant_probes", [])
    top_k = eval_cfg.get("top_k", 10)

    print(f"[1/4] Retrieval metrics ({len(cases)} kasus)...")
    retrieval = {
        "vector-only": evaluate_retrieval_metrics(vector_only, cases, top_k),
        "hybrid": evaluate_retrieval_metrics(retriever, cases, top_k),
    }
    print(f"      {retrieval}")

    print(f"[2/4] Chat end-to-end + judge ({len(cases)} kasus, LLM asli)...")
    try:
        chat_rows = evaluate_chat(responder, retriever, llm, cases)
    except LLMError as exc:
        raise SystemExit(f"Evaluasi chat gagal (LLM): {exc}") from exc

    print(f"[3/4] Refusal probes ({len(probes)})...")
    refusal_rows = evaluate_refusal(responder, probes)

    print(f"[4/4] Insight end-to-end + judge ({len(INSIGHT_QUERIES)} fokus)...")
    insight_rows = evaluate_insight(generator, retriever, llm, INSIGHT_QUERIES)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm": f"{provider}:{getattr(llm, 'model', '?')}",
        "corpus": len(reviews),
    }
    report = render_report(retrieval, chat_rows, refusal_rows, insight_rows, meta)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"\nReport tertulis ke {output}")

    grounded = [r["groundedness"] for r in chat_rows if r["groundedness"] > 0]
    print(f"Chat groundedness: {statistics.mean(grounded):.2f}/5" if grounded else "judge gagal")
    print(f"Refusal: {sum(r['refused'] for r in refusal_rows)}/{len(refusal_rows)}")


if __name__ == "__main__":
    main()
