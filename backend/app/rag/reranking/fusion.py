from __future__ import annotations

from typing import Any

from backend.app.rag.reranking.bm25 import bm25_index

# Reciprocal Rank Fusion constant (standard default; low sensitivity to
# this value is the whole point of RRF over hand-tuned score weights).
RRF_K = 60


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank retrieval candidates with hybrid lexical + semantic fusion.

    Fuses two independent rankings with Reciprocal Rank Fusion (RRF):

      - dense rank: FAISS cosine similarity to the query embedding
        (assumed to already be present as each candidate's `similarity`)
      - lexical rank: BM25 score against the query's exact terms

    RRF needs no score normalization or tuned weights, which matters
    here because the two signals live on very different scales (cosine
    similarity in [0, 1] vs. unbounded BM25 scores). Shared by the
    LangGraph reranker node and the offline RAG evaluation script so
    both score evidence identically.

    Deduplicates by chunk_id (falling back to text) before scoring, and
    mutates+returns each surviving candidate dict with `rerank_score`,
    `bm25_score`, `rank`, and `citation_id` set.
    """

    unique_candidates: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()

    for item in candidates:
        chunk_id = item.get("chunk_id")
        identity = chunk_id or item.get("text", "").strip()

        if not identity or identity in seen_chunks:
            continue

        seen_chunks.add(identity)
        unique_candidates.append(dict(item))

    if not unique_candidates:
        return []

    dense_ranked = sorted(
        unique_candidates,
        key=lambda item: float(item.get("similarity", 0.0)),
        reverse=True,
    )
    dense_rank = {id(item): rank for rank, item in enumerate(dense_ranked, start=1)}

    try:
        bm25_scores = bm25_index.score_by_chunk_id(query)
    except FileNotFoundError:
        bm25_scores = {}

    lexical_ranked = sorted(
        unique_candidates,
        key=lambda item: bm25_scores.get(item.get("chunk_id"), 0.0),
        reverse=True,
    )
    lexical_rank = {id(item): rank for rank, item in enumerate(lexical_ranked, start=1)}

    for item in unique_candidates:
        d_rank = dense_rank.get(id(item), len(unique_candidates) + 1)
        l_rank = lexical_rank.get(id(item), len(unique_candidates) + 1)
        item["rerank_score"] = 1.0 / (RRF_K + d_rank) + 1.0 / (RRF_K + l_rank)
        item["bm25_score"] = bm25_scores.get(item.get("chunk_id"), 0.0)

    ranked = sorted(
        unique_candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )[:top_n]

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["citation_id"] = f"[E{rank}]"

    return ranked
