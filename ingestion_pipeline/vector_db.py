from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Support both running as a module and executing directly as a script.
try:
    from ingestion_pipeline.pdf_ingest import load_and_chunk_documents
except ModuleNotFoundError:
    from pdf_ingest import load_and_chunk_documents

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_vectorstore = None


def get_vector_store():
    """Return a singleton Chroma vector store instance.
    
    The vector store is lazy-initialized. On first call, it loads the persisted
    database from disk (if it exists). If the database doesn't exist, an empty
    one will be created.
    """

    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    return _vectorstore


def vector_store_is_empty():
    """Check if the vector store is empty."""
    vectorstore = get_vector_store()
    count = vectorstore._collection.count()
    return count == 0


if __name__ == "__main__":
    # For manual testing/initialization only
    import sys

    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        chunks = load_and_chunk_documents(source_path)
        
        if chunks:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
            vectorstore.persist()
            print(f"Vector DB created successfully with {len(chunks)} chunks")
            print(f"Total vectors in database: {vectorstore._collection.count()}")
        else:
            print("No documents to add to vector store")
    else:
        print("Usage: python vector_db.py <path_to_file_or_folder>")
        print("\nExample: python vector_db.py essays/")
        print("Example: python vector_db.py uploads/document.pdf")