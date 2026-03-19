"""
Answer generation using the LLM with document context.
"""

from config import get_llm


def generate_answer(query: str, chunks: list[str]) -> str:
    """Generate a final answer using retrieved context.
    
    Args:
        query: The user's question.
        chunks: Retrieved document chunks to use as context.
    
    Returns:
        The generated answer.
    """
    llm = get_llm()
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content
