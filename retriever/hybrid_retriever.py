from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi

from retriever.query_expansion import generate_queries
from retriever.reranker import rerank

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

data = vectorstore._collection.get()

documents = data["documents"]

tokenized_docs = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

def hybrid_search(query, k=5):
    vector_results = vectorstore.similarity_search(query, k=k)

    vector_docs = [doc.page_content for doc in vector_results]

    # BM25 search
    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [documents[i] for i in bm25_indices]

    # Merge results
    combined = list(set(vector_docs + bm25_docs))

    return combined[:k]

def multiquery_hybrid_search(query):

    # Step 1: generate alternative queries
    queries = generate_queries(query)

    all_results = []

    # Step 2: run hybrid retrieval for each query
    for q in queries:

        results = hybrid_search(q, k=20)   # retrieve more candidates

        all_results.extend(results)

    # Step 3: remove duplicates
    unique_results = list(set(all_results))

    # Step 4: rerank and keep top 5
    reranked_results = rerank(query, unique_results, top_k=5)

    return reranked_results
