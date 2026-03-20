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

    prompt = f"""
    You are a knowledgeable assistant.

    Answer the question in a detailed and structured way using the provided context.

    Instructions:
    - Provide a clear and complete explanation
    - Use multiple sentences (at least 5-8 lines if possible)
    - Include reasoning and explanation, not just the final answer
    - If applicable, break the answer into paragraphs or bullet points
    - Do NOT be overly brief

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content
