from generation.pipeline import rag_pipeline

while True:

    query = input("\nQuestion: ")

    answer, chunks = rag_pipeline(query)

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieved Context:\n")

    for c in chunks:
        print("-", c[:200])