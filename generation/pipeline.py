"""
RAG pipeline coordinating retrieval and generation stages.
"""

from retriever.hybrid_retriever import multiquery_hybrid_search
from generation.generator import generate_answer


def rag_pipeline(query: str) -> tuple[str, list[str]]:
    """Execute the RAG pipeline: retrieve context then generate answer.
    
    Args:
        query: User's question.
    
    Returns:
        Tuple of (answer, source_chunks).
    """
    chunks = multiquery_hybrid_search(query)
    answer = generate_answer(query, chunks)
    return answer, chunks
