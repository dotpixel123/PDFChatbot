from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from pdf_ingest import chunk_pdf

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_vectorstore = None


def get_vector_store():
    """Return a singleton Chroma vector store instance."""

    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    return _vectorstore


def create_vector_db():

    chunks = chunk_pdf()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()

    print("Vector DB created successfully")
    print("Total vectors:", vectorstore._collection.count())

    data = vectorstore._collection.get()

    print("Keys:", data.keys())
    print("Number of docs:", len(data["documents"]))

    print("\nFirst stored chunk:")
    print(data["documents"][0])

    print("\nMetadata:")
    print(data["metadatas"][0])

    return vectorstore


if __name__ == "__main__":
    create_vector_db()