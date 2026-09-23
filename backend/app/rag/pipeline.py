
from __future__ import annotations

from backend.app.rag.citations.citation_builder import (
    build_citations,
)

from backend.app.rag.chunking.chunker import (
    chunk_documents,
)

from backend.app.rag.ingestion.loader import (
    load_knowledge_base,
)

from backend.app.rag.retrieval.vector_store import (
    VectorStore,
)


class RAGPipeline:

    def __init__(self):

        self.vector_store = VectorStore()

    # ========================================================
    # INGEST
    # ========================================================

    def ingest(self):

        print("=" * 70)
        print(
            "NEUROSCREEN AI - RAG INGESTION"
        )
        print("=" * 70)

        documents = load_knowledge_base()

        if not documents:

            raise RuntimeError(
                "No knowledge-base documents found."
            )

        print()
        print(
            f"Documents loaded: "
            f"{len(documents)}"
        )

        chunks = chunk_documents(
            documents
        )

        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )

        self.vector_store.build(
            chunks
        )

        print()
        print(
            "RAG ingestion completed."
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:

        results = self.vector_store.search(
            query=query,
            top_k=top_k,
        )

        citations = build_citations(
            results
        )

        return {
            "query": query,
            "results": results,
            "citations": citations,
        }


rag_pipeline = RAGPipeline()