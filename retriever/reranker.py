from sentence_transformers import CrossEncoder


# load reranker model
reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank(query, documents, top_k=5):
    """
    Rerank retrieved documents using a cross-encoder model
    """

    # create query-document pairs
    pairs = [(query, doc) for doc in documents]

    # compute relevance scores
    scores = reranker.predict(pairs)

    # combine documents with scores
    scored_docs = list(zip(documents, scores))

    # sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # select top results
    top_docs = [doc for doc, score in scored_docs[:top_k]]

    return top_docs