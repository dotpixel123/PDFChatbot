from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_pdf


def load_and_chunk_documents(file_path: str):
    """Load a PDF file and chunk it into text segments for vector ingestion."""

    documents = load_pdf(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks for {file_path}")

    return chunks


def chunk_pdf():
    """Legacy entrypoint for the hardcoded meditation.pdf file."""
    return load_and_chunk_documents("meditation.pdf")
