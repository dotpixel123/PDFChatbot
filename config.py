"""
Shared configuration and utilities for the RAG system.
Centralizes model initialization, API keys, and constants.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Document Processing
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval Configuration
BM25_K = 60
HYBRID_TOP_N = 20
RERANK_TOP_K = 5
QUERY_EXPANSION_COUNT = 3

# Storage
VECTOR_DB_DIR = "chroma_db"
UPLOAD_DIR = "uploads"
REGISTRY_FILE = "document_registry.json"


def get_llm():
    """Get or create the LLM instance (lazy singleton pattern)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE
    )
