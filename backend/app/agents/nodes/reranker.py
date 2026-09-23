
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState
from backend.app.rag.citations.citation_builder import build_citations
from backend.app.rag.reranking.fusion import rerank_candidates

MAX_EVIDENCE = 5


def reranker_node(state: NeuroScreenState) -> dict:
    """
    Rerank retrieved RAG evidence with hybrid lexical + semantic fusion.

    The retrieval node over-fetches a wide candidate pool using FAISS
    dense similarity alone. This node re-scores that pool with
    Reciprocal Rank Fusion over dense similarity and BM25 lexical
    scoring (see `backend.app.rag.reranking.fusion`), which tends to
    surface documents containing the query's actual keywords rather
    than only generic paraphrase-level matches.
    """

    evidence = state.get("retrieved_evidence", [])

    if not evidence:
        return {
            "retrieved_evidence": [],
            "citations": [],
            "errors": ["Reranker received no retrieved evidence."],
        }

    query = state.get("query", "")

    ranked_evidence = rerank_candidates(query, evidence, top_n=MAX_EVIDENCE)
    citations = build_citations(ranked_evidence)

    return {
        "retrieved_evidence": ranked_evidence,
        "citations": citations,
    }
