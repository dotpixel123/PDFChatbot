import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents_from_path(path: str):
    """Load all supported documents (.pdf, .md, .txt) from a file or folder.
    
    Args:
        path: Path to a single file or a folder containing multiple files.
    
    Returns:
        List of loaded documents.
    """

    all_documents = []

    # Handle single file
    if os.path.isfile(path):
        file_paths = [path]
    # Handle directory
    elif os.path.isdir(path):
        file_paths = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
    else:
        print(f"Error: Path does not exist: {path}")
        return all_documents

    for file_path in file_paths:
        try:
            # PDF files
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_documents.extend(docs)

            # Markdown / text files
            elif file_path.lower().endswith((".md", ".txt")):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                all_documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue

    print(f"Loaded {len(all_documents)} documents from {path}")

    return all_documents