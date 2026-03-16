from langchain_community.document_loaders import PyPDFLoader

def load_pdf():
    file_path = "meditation.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    return documents