"""Pluggable searchers used by the RAG pipeline and evaluations.

Every searcher implements `search(query, num_results) -> list[dict]`.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class TextSearcher:
    """Wraps a fitted minsearch Index with query-time field boosts."""

    def __init__(self, index, documents: list[dict[str, Any]], boost: dict[str, float] | None = None):
        self.index = index
        self.documents = documents
        self.boost = boost or {"name": 3.0, "taxa": 2.0, "study_objective": 1.5}

    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        return self.index.search(query, num_results=num_results, boost_dict=self.boost)


class VectorSearcher:
    """Cosine similarity over normalized embeddings stored in-memory."""

    def __init__(
        self,
        embeddings: np.ndarray,
        documents: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        from sentence_transformers import SentenceTransformer

        self.embeddings = embeddings
        self.documents = documents
        self.model = SentenceTransformer(model_name)

    def embed(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]

    def _top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.embeddings @ query_vec
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[int(i)])) for i in idx]

    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        q = self.embed(query)
        pairs = self._top_k(q, num_results)
        return [self.documents[i] for i, _ in pairs]


def reciprocal_rank_fusion(
    rankings: Iterable[list[dict[str, Any]]],
    key: str = "study_id",
    k: int = 60,
) -> list[dict[str, Any]]:
    """Combine multiple ranked lists via RRF, dedup by `key`."""
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for ranked in rankings:
        for rank, doc in enumerate(ranked):
            doc_key = str(doc.get(key))
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(doc_key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [docs[doc_key] for doc_key, _ in ordered]


class HybridSearcher:
    """RRF fusion of a text searcher and a vector searcher."""

    def __init__(
        self,
        text: TextSearcher,
        vector: VectorSearcher,
        fetch_k: int = 20,
    ):
        self.text = text
        self.vector = vector
        self.fetch_k = fetch_k

    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        t = self.text.search(query, num_results=self.fetch_k)
        v = self.vector.search(query, num_results=self.fetch_k)
        fused = reciprocal_rank_fusion([t, v])
        return fused[:num_results]


class RerankedSearcher:
    """Wrap any base searcher with a cross-encoder rerank stage."""

    def __init__(
        self,
        base,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        fetch_k: int = 20,
    ):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.fetch_k = fetch_k
        self.model = CrossEncoder(model_name)

    def _doc_text(self, doc: dict[str, Any]) -> str:
        fields = [doc.get("name"), doc.get("taxa"), doc.get("study_objective"), doc.get("citation")]
        return " | ".join(f for f in fields if f)

    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        candidates = self.base.search(query, num_results=self.fetch_k)
        if not candidates:
            return []
        pairs = [(query, self._doc_text(doc)) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda kv: kv[0], reverse=True)]
        return ranked[:num_results]


REWRITE_INSTRUCTIONS = """
Rewrite the user's question into a concise search query for a wildlife
tracking studies index. Expand common bird/animal names to include the
scientific / taxon name where you are confident (e.g. "turkey vulture"
-> "turkey vulture Cathartes aura"). Return ONLY the rewritten query.
""".strip()


class QueryRewriter:
    """LLM-backed query rewriter with best-effort fallback to the original query."""

    def __init__(self, llm_client, model: str | None = None):
        self.llm_client = llm_client
        self.model = model or getattr(llm_client, "model", "gpt-4o-mini")

    def rewrite(self, query: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_INSTRUCTIONS},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            rewritten = (response.choices[0].message.content or "").strip().strip('"')
            return rewritten or query
        except Exception:
            return query


class RewritingSearcher:
    """Prepend a query-rewrite step to any base searcher."""

    def __init__(self, base, rewriter: QueryRewriter):
        self.base = base
        self.rewriter = rewriter
        self.last_rewrite: str | None = None

    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        rewritten = self.rewriter.rewrite(query)
        self.last_rewrite = rewritten
        return self.base.search(rewritten, num_results=num_results)
