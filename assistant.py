"""Factory that wires index + searcher + LLM into a RAGWithMetrics instance."""

from __future__ import annotations

import os
from functools import lru_cache

from ingest import load_documents, load_embeddings, load_text_index
from llmclient import get_llm_client
from metrics import RAGWithMetrics
from search import (
    HybridSearcher,
    QueryRewriter,
    RerankedSearcher,
    RewritingSearcher,
    TextSearcher,
    VectorSearcher,
)


COURSE = "wildlife-tracking-rag"


@lru_cache(maxsize=1)
def _load_assets():
    documents = load_documents()
    text_index = load_text_index()
    embeddings = load_embeddings()
    return documents, text_index, embeddings


def build_searcher(
    strategy: str = "hybrid_rerank_rewrite",
    llm_client=None,
):
    """Build the retrieval stack.

    Strategies:
      - "text"                 : minsearch text search only
      - "vector"               : dense vector search only
      - "hybrid"               : RRF(text, vector)
      - "hybrid_rerank"        : cross-encoder rerank on top of hybrid
      - "hybrid_rerank_rewrite": add LLM query rewriting on top of hybrid+rerank
    """
    documents, text_index, embeddings = _load_assets()

    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    text = TextSearcher(text_index, documents)
    vector = VectorSearcher(embeddings, documents, model_name=embed_model)

    if strategy == "text":
        return text
    if strategy == "vector":
        return vector

    hybrid = HybridSearcher(text, vector)
    if strategy == "hybrid":
        return hybrid

    reranked = RerankedSearcher(hybrid)
    if strategy == "hybrid_rerank":
        return reranked

    if strategy == "hybrid_rerank_rewrite":
        rewriter = QueryRewriter(llm_client or get_llm_client())
        return RewritingSearcher(reranked, rewriter)

    raise ValueError(f"Unknown retrieval strategy: {strategy}")


def create_assistant(strategy: str | None = None) -> RAGWithMetrics:
    strategy = strategy or os.getenv("RAG_STRATEGY", "hybrid_rerank_rewrite")
    llm_client = get_llm_client()
    searcher = build_searcher(strategy=strategy, llm_client=llm_client)
    return RAGWithMetrics(search_engine=searcher, llm_client=llm_client)


if __name__ == "__main__":
    import sys

    assistant = create_assistant()
    query = " ".join(sys.argv[1:]) or "Which studies track turkey vultures?"
    print("Q:", query)
    answer = assistant.rag(query)
    print("A:", answer)
