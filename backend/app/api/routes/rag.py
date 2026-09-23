from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.rag.pipeline import rag_pipeline
from backend.app.schemas.rag import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router.post("/retrieve", response_model=RetrievalResponse)
def retrieve_evidence(request: RetrievalRequest) -> RetrievalResponse:
    """Direct evidence retrieval, independent of the screening workflow.

    Useful for the evidence-explorer UI and for debugging retrieval
    quality without running a full assessment.
    """

    try:
        result = rag_pipeline.retrieve(query=request.query, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"RAG index unavailable: {exc}") from exc

    return RetrievalResponse(
        query=result["query"],
        evidence=result["results"],
        citations=result["citations"],
    )
