
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState
from backend.app.rag.pipeline import rag_pipeline


def retrieval_node(state: NeuroScreenState) -> dict:
    """
    Retrieve evidence from the validated knowledge base.
    """

    query = state.get("query")

    if not query:
        return {
            "retrieved_evidence": [],
            "citations": [],
            "errors": [
                "Retrieval node received an empty query."
            ],
        }

    try:
        # Over-fetch a wider candidate pool so the reranker node has
        # enough documents to meaningfully re-score and reorder before
        # trimming down to the final top-N evidence set.
        result = rag_pipeline.retrieve(
            query=query,
            top_k=20,
        )

        return {
            "retrieved_evidence": result.get(
                "results",
                []
            ),
            "citations": result.get(
                "citations",
                []
            ),
        }

    except Exception as exc:
        return {
            "retrieved_evidence": [],
            "citations": [],
            "errors": [
                f"Retrieval node failed: {str(exc)}"
            ],
        }