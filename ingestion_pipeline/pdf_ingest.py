"""
Document loading and chunking for ingestion.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from .loader import load_documents_from_path


def load_and_chunk_documents(source: str) -> list:
    """Load and chunk documents for vector ingestion.
    
    Args:
        source: Path to a file or folder containing documents.
    
    Returns:
        List of chunked documents ready for embedding.
    """
    documents = load_documents_from_path(source)

    if not documents:
        print(f"Warning: No documents found at {source}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    return chunks

