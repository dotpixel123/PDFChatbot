from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_pdf

def chunk_pdf():

    documents = load_pdf()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks
