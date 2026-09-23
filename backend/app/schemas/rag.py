from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    model_config = {"extra": "allow"}

    chunk_id: str | None = None
    citation_id: str | None = None
    source: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    chunk_index: int | None = None
    text: str
    similarity: float | None = None
    rerank_score: float | None = None
    rank: int | None = None


class Citation(BaseModel):
    citation_id: str
    source: str | None = None
    file_name: str | None = None
    chunk_id: str | None = None
    similarity: float | None = None
    text: str


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResponse(BaseModel):
    query: str
    evidence: list[EvidenceItem]
    citations: list[Citation]
