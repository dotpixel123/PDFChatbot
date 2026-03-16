from retriever.hybrid_retriever import multiquery_hybrid_search
from generation.generator import generate_answer


def rag_pipeline(query):

    # retrieval
    chunks = multiquery_hybrid_search(query)

    # generation
    answer = generate_answer(query, chunks)

    return answer, chunks