"""Offline retrieval evaluation for all strategies (no LLM required for the
base variants). Emits `data/retrieval_eval.csv` and prints a summary.

Metrics
-------
Hit@k : the gold study id appears in the top-k results
MRR   : mean reciprocal rank of the gold study id (0 if not retrieved)

The `hybrid_rerank_rewrite` variant needs an LLM for query rewriting. We only
evaluate it if `OPENAI_API_KEY` is set; otherwise it is skipped and a note is
printed.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest import load_documents, load_embeddings, load_text_index
from search import (
    HybridSearcher,
    QueryRewriter,
    RerankedSearcher,
    RewritingSearcher,
    TextSearcher,
    VectorSearcher,
)


def evaluate(searcher, ground_truth: list[dict], k: int = 5) -> dict[str, float]:
    hits = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    for row in ground_truth:
        results = searcher.search(row["question"], num_results=k)
        ids = [r.get("study_id") for r in results]
        target = row["study_id"]
        for kk in (1, 3, 5):
            if target in ids[:kk]:
                hits[kk] += 1
        if target in ids:
            mrr += 1.0 / (ids.index(target) + 1)
    n = len(ground_truth) or 1
    return {
        "Hit@1": hits[1] / n,
        "Hit@3": hits[3] / n,
        "Hit@5": hits[5] / n,
        "MRR": mrr / n,
    }


def main() -> None:
    documents = load_documents()
    text_index = load_text_index()
    embeddings = load_embeddings()

    gt_path = Path("data/ground_truth.json")
    ground_truth = json.loads(gt_path.read_text())
    print(f"Evaluating {len(ground_truth)} questions over {len(documents)} studies")

    text = TextSearcher(text_index, documents)
    vector = VectorSearcher(embeddings, documents)
    hybrid = HybridSearcher(text, vector)
    reranked = RerankedSearcher(hybrid)

    strategies = [
        ("text", text),
        ("vector", vector),
        ("hybrid", hybrid),
        ("hybrid_rerank", reranked),
    ]

    if os.getenv("OPENAI_API_KEY"):
        from llmclient import get_llm_client

        rewritten = RewritingSearcher(reranked, QueryRewriter(get_llm_client()))
        strategies.append(("hybrid_rerank_rewrite", rewritten))
    else:
        print("(skipping hybrid_rerank_rewrite: OPENAI_API_KEY not set)")

    results = {}
    for name, searcher in strategies:
        print(f"  {name}...", end=" ", flush=True)
        results[name] = evaluate(searcher, ground_truth)
        print(results[name])

    out_path = Path("data/retrieval_eval.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "Hit@1", "Hit@3", "Hit@5", "MRR"])
        for name, m in results.items():
            writer.writerow([name, f"{m['Hit@1']:.3f}", f"{m['Hit@3']:.3f}", f"{m['Hit@5']:.3f}", f"{m['MRR']:.3f}"])
    print(f"\nWrote {out_path}")

    best = max(results.items(), key=lambda kv: kv[1]["MRR"])[0]
    print(f"Best strategy by MRR: {best}")


if __name__ == "__main__":
    main()
