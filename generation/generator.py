from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2
)


def generate_answer(query, chunks):
    """
    Generate final answer using retrieved chunks as context
    """

    context = "\n\n".join(chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content