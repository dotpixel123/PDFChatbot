from generation.pipeline import rag_pipeline
from ingestion_pipeline.vector_db import vector_store_is_empty


def main():
    """CLI for the RAG system with document management."""
    
    # Check if database is empty on startup
    if vector_store_is_empty():
        print("\n" + "="*60)
        print("RAG System - Vector database is empty")
        print("="*60)
        print("\nTo use the system, you must first add documents to the database.")
        print("Please use the /upload endpoint in the API or add documents manually.")
        print("\nExample (if using API):")
        print("  curl -X POST -F 'file=@your_document.pdf' http://localhost:8000/upload")
        print("\nOr run: uv run main_api.py")
        print("Then access http://localhost:8000/docs for the web interface.\n")
        return

    # Main query loop
    print("\n" + "="*60)
    print("RAG System - Ready to answer your questions")
    print("="*60)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            answer, chunks = rag_pipeline(query)

            print("\nAnswer:")
            print(answer)

            print("\nRetrieved Context:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n[{i}] {chunk[:200]}...")
            print()
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
