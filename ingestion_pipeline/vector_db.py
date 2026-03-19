"""
Vector database initialization and management using Chroma.
Handles persistence and lazy initialization of the vector store.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import VECTOR_DB_DIR, EMBEDDING_MODEL

try:
    from ingestion_pipeline.pdf_ingest import load_and_chunk_documents
except ModuleNotFoundError:
    from pdf_ingest import load_and_chunk_documents

_vectorstore = None


def get_vector_store():
    """Get or create the vector store (lazy singleton)."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
    return _vectorstore


def vector_store_is_empty() -> bool:
    """Check if the vector store has any documents."""
    vectorstore = get_vector_store()
    return vectorstore._collection.count() == 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        chunks = load_and_chunk_documents(source_path)
        
        if chunks:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=VECTOR_DB_DIR
            )
            vectorstore.persist()
            print(f"✓ Vector DB created with {len(chunks)} chunks")
        else:
            print("✗ No documents found")
    else:
        print("Usage: uv run ingestion_pipeline/vector_db.py <path>")