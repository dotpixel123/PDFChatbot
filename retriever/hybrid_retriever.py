"""
Hybrid retrieval combining vector search and BM25 ranking.
Improves recall and relevance through multi-stage retrieval.
"""

from typing import List
from rank_bm25 import BM25Okapi

from config import BM25_K, HYBRID_TOP_N, RERANK_TOP_K
from ingestion_pipeline.vector_db import get_vector_store
from retriever.query_expansion import generate_queries
from retriever.reranker import rerank

_bm25 = None
_bm25_docs: List[str] = []
_bm25_dirty = True


def mark_bm25_dirty():
    """Mark BM25 index as stale (call after adding/removing documents)."""
    global _bm25_dirty
    _bm25_dirty = True


def _build_bm25():
    """Build or rebuild the BM25 index from the current vector store."""
    global _bm25, _bm25_docs, _bm25_dirty

    vectorstore = get_vector_store()
    data = vectorstore._collection.get()
    documents = data.get("documents", [])

    if not documents:
        _bm25 = None
        _bm25_docs = []
        _bm25_dirty = False
        return

    tokenized_docs = [doc.split() for doc in documents]
    _bm25 = BM25Okapi(tokenized_docs)
    _bm25_docs = documents
    _bm25_dirty = False


def _ensure_bm25():
    """Rebuild BM25 if marked dirty."""
    if _bm25_dirty or _bm25 is None:
        _build_bm25()


def rrf_merge(lists: List[List[str]], k: int = BM25_K, top_n: int = HYBRID_TOP_N) -> List[str]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for lst in lists:
        for rank, doc in enumerate(lst):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked][:top_n]


def hybrid_search(query: str, k: int = 5) -> List[str]:
    """Perform hybrid retrieval: vector search + BM25 + RRF merge."""
    _ensure_bm25()

    vectorstore = get_vector_store()
    vector_results = vectorstore.similarity_search(query, k=k)
    vector_docs = [doc.page_content for doc in vector_results]

    if _bm25 is None or not _bm25_docs:
        return vector_docs[:k]

    tokenized_query = query.split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [_bm25_docs[i] for i in bm25_indices]
    merged = rrf_merge([vector_docs, bm25_docs], k=BM25_K, top_n=k * 2)
    return merged[:k]


def multiquery_hybrid_search(query: str) -> List[str]:
    """Multi-stage retrieval: expand queries, retrieve, rerank."""
    queries = generate_queries(query)

    all_results: List[str] = []
    for q in queries:
        results = hybrid_search(q, k=20)
        all_results.extend(results)

    unique_results = list(dict.fromkeys(all_results))
    candidates = unique_results[:30]
    reranked_results = rerank(query, candidates, top_k=RERANK_TOP_K)

    return reranked_results