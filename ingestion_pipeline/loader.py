from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str = "meditation.pdf"):
    """Load a PDF file and return LangChain Document objects."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from {file_path}")

    return documents