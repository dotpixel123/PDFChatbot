import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents_from_folder(folder_path: str):
    """
    Load all supported documents (.pdf, .md, .txt) from a folder.
    """

    all_documents = []

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        # PDF files
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_documents.extend(docs)

        # Markdown / text files
        elif file.endswith(".md") or file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            all_documents.extend(docs)

    print(f"Loaded {len(all_documents)} documents from {folder_path}")

    return all_documents