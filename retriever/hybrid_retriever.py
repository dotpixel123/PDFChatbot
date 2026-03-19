from typing import List

from rank_bm25 import BM25Okapi

from ingestion_pipeline.vector_db import get_vector_store
from retriever.query_expansion import generate_queries
from retriever.reranker import rerank


# ----------------------
# BM25 (lazy + rebuild)
# ----------------------
_bm25 = None
_bm25_docs: List[str] = []
_bm25_dirty = True


def mark_bm25_dirty():
    """Mark BM25 index as stale.

    Call this after adding or removing documents from the vector store.
    """

    global _bm25_dirty
    _bm25_dirty = True


def _build_bm25():
    """Build or rebuild the BM25 index from the current vector store."""

    global _bm25, _bm25_docs, _bm25_dirty

    vectorstore = get_vector_store()
    data = vectorstore._collection.get()
    documents = data.get("documents", [])

    tokenized_docs = [doc.split() for doc in documents]

    _bm25 = BM25Okapi(tokenized_docs)
    _bm25_docs = documents
    _bm25_dirty = False


def _ensure_bm25():
    if _bm25_dirty or _bm25 is None:
        _build_bm25()


# ----------------------
# Ranking helpers
# ----------------------

def rrf_merge(lists: List[List[str]], k: int = 60, top_n: int = 20) -> List[str]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""

    scores = {}
    for lst in lists:
        for rank, doc in enumerate(lst):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked][:top_n]


# ----------------------
# Retrieval functions
# ----------------------

def hybrid_search(query: str, k: int = 5) -> List[str]:
    """Perform hybrid retrieval using vector search + BM25, then merge via RRF."""

    _ensure_bm25()

    vectorstore = get_vector_store()
    vector_results = vectorstore.similarity_search(query, k=k)
    vector_docs = [doc.page_content for doc in vector_results]

    # BM25 ranking
    tokenized_query = query.split()
    bm25_scores = _bm25.get_scores(tokenized_query)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [_bm25_docs[i] for i in bm25_indices]

    # Merge both ranked lists
    merged = rrf_merge([vector_docs, bm25_docs], k=60, top_n=k * 2)
    return merged[:k]


def multiquery_hybrid_search(query: str) -> List[str]:
    """Generate multiple query variants, retrieve candidates, then rerank."""

    queries = generate_queries(query)

    all_results: List[str] = []
    for q in queries:
        results = hybrid_search(q, k=20)
        all_results.extend(results)

    # Preserve order while removing duplicates
    unique_results = list(dict.fromkeys(all_results))

    # Rerank candidates with cross-encoder
    candidates = unique_results[:30]
    reranked_results = rerank(query, candidates, top_k=5)

    return reranked_results
