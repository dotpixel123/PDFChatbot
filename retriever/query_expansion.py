from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

def generate_queries(question, n_queries=3):

    prompt = f"""
Generate {n_queries} alternative search queries for the following question.
Each query should capture the same intent but use different wording.

Question: {question}

Return each query on a new line.
"""

    response = llm.invoke(prompt)

    queries = response.content.split("\n")

    return [q.strip("- ").strip() for q in queries if q.strip()]
