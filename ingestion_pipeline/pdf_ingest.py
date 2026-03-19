from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader import load_documents_from_folder


def load_and_chunk_documents():
    """Load a PDF file and chunk it into text segments for vector ingestion."""

    documents = load_documents_from_folder("essays")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)}")

    return chunks

