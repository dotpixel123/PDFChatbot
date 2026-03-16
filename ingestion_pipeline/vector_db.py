from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from pdf_ingest import chunk_pdf

def create_vector_db():

    chunks = chunk_pdf()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
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