from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loader import load_documents_from_path


def load_and_chunk_documents(source: str):
    documents = load_documents_from_path(source)
    if not documents:
        print(f'Warning: No documents found at {source}')
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f'Created {len(chunks)} chunks from {len(documents)} documents')
    return chunks
