"""
Query expansion for improving retrieval coverage.
Generates multiple query variants to search for the same answer.
"""

from config import get_llm, QUERY_EXPANSION_COUNT


def generate_queries(question: str, n_queries: int = QUERY_EXPANSION_COUNT) -> list[str]:
    """Generate alternative search queries.
    
    Args:
        question: The original user question.
        n_queries: Number of alternative queries to generate.
    
    Returns:
        List of expanded queries.
    """
    llm = get_llm()

    prompt = f"""Generate {n_queries} alternative search queries for the following question.
Each query should capture the same intent but use different wording.

Question: {question}

Return each query on a new line, without numbering or bullets."""

    response = llm.invoke(prompt)
    queries = response.content.split("\n")
    return [q.strip() for q in queries if q.strip()]

