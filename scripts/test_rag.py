from backend.app.rag.pipeline import rag_pipeline


def main():
    print("=" * 70)
    print("NEUROSCREEN AI - RAG RETRIEVAL TEST")
    print("=" * 70)

    query = (
        "How is autism spectrum disorder screened in children, "
        "and what happens after a positive screening result?"
    )

    print()
    print(f"Query: {query}")
    print()

    response = rag_pipeline.retrieve(
        query=query,
        top_k=5,
    )

    print("=" * 70)
    print("RETRIEVED RESULTS")
    print("=" * 70)

    for i, result in enumerate(response["results"], start=1):
        print()
        print(f"[RESULT {i}]")
        print(f"Source: {result.get('source')}")
        print(f"File: {result.get('file_name')}")
        print(f"Chunk: {result.get('chunk_id')}")
        print(f"Similarity: {result.get('similarity'):.4f}")

        text = result.get("text", "").replace("\n", " ")
        print(f"Text: {text[:500]}")

    print()
    print("=" * 70)
    print("CITATIONS")
    print("=" * 70)

    for citation in response["citations"]:
        print(citation)

    print()
    print("=" * 70)
    print("RAG RETRIEVAL TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()