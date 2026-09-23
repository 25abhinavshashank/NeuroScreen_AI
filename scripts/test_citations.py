
from backend.app.rag.citations.citation_builder import build_citations

test_documents = [
    {
        "source": "CDC",
        "file_name": "cdc_screening.pdf",
        "chunk_id": "chunk_001",
        "similarity": 0.91,
        "text": (
            "Developmental screening can help identify developmental "
            "concerns and determine whether further evaluation may be appropriate."
        ),
    }
]

citations = build_citations(test_documents)

print("\nCITATIONS:")
print(citations)