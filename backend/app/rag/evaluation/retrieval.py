from __future__ import annotations

import math
import posixpath
from typing import Any


def get_chunk_id(item: dict[str, Any]) -> str | None:
    """
    Extract the chunk ID from a retrieved RAG item.
    """

    metadata = item.get("metadata", {}) or {}

    return (
        item.get("chunk_id")
        or metadata.get("chunk_id")
        or item.get("id")
        or metadata.get("id")
    )


def get_source(item: dict[str, Any]) -> str | None:
    """
    Extract the source document's base file name.

    Chunk metadata stores `source` as a full filesystem path (e.g.
    "knowledge_base\\guidelines\\CDC_autism_screening.pdf") while
    evaluation datasets reference documents by base file name (e.g.
    "CDC_autism_screening.pdf"). Prefer `file_name` and, when only a
    path is available, normalize both `/` and `\\` separators to the
    base name so the two agree instead of silently never matching.
    """

    metadata = item.get("metadata", {}) or {}

    raw = (
        item.get("file_name")
        or metadata.get("file_name")
        or item.get("source_name")
        or metadata.get("source_name")
        or item.get("source")
        or metadata.get("source")
    )

    if not raw:
        return None

    return posixpath.basename(str(raw).replace("\\", "/"))


def reciprocal_rank(
    retrieved: list[dict[str, Any]],
    relevant_chunks: set[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    RR = 1 / rank of the first relevant retrieved chunk.
    """

    for rank, item in enumerate(retrieved, start=1):
        chunk_id = get_chunk_id(item)

        if chunk_id in relevant_chunks:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved: list[dict[str, Any]],
    relevant_chunks: set[str],
) -> float:
    """
    Calculate Recall@K.

    Recall = relevant retrieved chunks / total relevant chunks.
    """

    if not relevant_chunks:
        return 0.0

    retrieved_ids = {
        get_chunk_id(item)
        for item in retrieved
    }

    retrieved_ids.discard(None)

    hits = retrieved_ids.intersection(relevant_chunks)

    return len(hits) / len(relevant_chunks)


def precision_at_k(
    retrieved: list[dict[str, Any]],
    relevant_chunks: set[str],
) -> float:
    """
    Calculate Precision@K.

    Precision = relevant retrieved chunks / retrieved chunks.
    """

    if not retrieved:
        return 0.0

    retrieved_ids = [
        get_chunk_id(item)
        for item in retrieved
    ]

    retrieved_ids = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id is not None
    ]

    if not retrieved_ids:
        return 0.0

    hits = sum(
        chunk_id in relevant_chunks
        for chunk_id in retrieved_ids
    )

    return hits / len(retrieved_ids)


def ndcg_at_k(
    retrieved: list[dict[str, Any]],
    relevant_chunks: set[str],
) -> float:
    """
    Calculate NDCG@K with binary relevance.

    DCG = sum(rel_i / log2(i + 1)) over the retrieved ranking.
    IDCG = DCG of the ideal ranking (all relevant chunks first).
    NDCG = DCG / IDCG, defined as 0.0 when there are no relevant chunks.
    """

    if not relevant_chunks:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(retrieved, start=1):
        chunk_id = get_chunk_id(item)
        relevance = 1.0 if chunk_id in relevant_chunks else 0.0
        if relevance:
            dcg += relevance / math.log2(rank + 1)

    ideal_hits = min(len(relevant_chunks), len(retrieved))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_single_query(
    retrieved: list[dict[str, Any]],
    relevant_chunks: list[str],
    relevant_sources: list[str] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """
    Evaluate one retrieval query.
    """

    top_k = retrieved[:k]

    relevant_chunk_set = set(relevant_chunks)
    relevant_source_set = set(relevant_sources or [])

    retrieved_chunk_ids = [
        get_chunk_id(item)
        for item in top_k
    ]

    retrieved_sources = [
        get_source(item)
        for item in top_k
    ]

    chunk_hits = [
        chunk_id
        for chunk_id in retrieved_chunk_ids
        if chunk_id in relevant_chunk_set
    ]

    source_hits = [
        source
        for source in retrieved_sources
        if source in relevant_source_set
    ]

    return {
        "k": k,
        "retrieved_count": len(top_k),
        "retrieved_chunks": retrieved_chunk_ids,
        "retrieved_sources": retrieved_sources,
        "relevant_chunks": list(relevant_chunks),
        "relevant_sources": list(relevant_sources or []),
        "chunk_hits": chunk_hits,
        "source_hits": source_hits,
        "recall_at_k": recall_at_k(
            top_k,
            relevant_chunk_set,
        ),
        "precision_at_k": precision_at_k(
            top_k,
            relevant_chunk_set,
        ),
        "reciprocal_rank": reciprocal_rank(
            top_k,
            relevant_chunk_set,
        ),
        "ndcg_at_k": ndcg_at_k(
            top_k,
            relevant_chunk_set,
        ),
    }


def evaluate_dataset(
    evaluations: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """
    Evaluate the complete RAG evaluation dataset.
    """

    results = []

    for case in evaluations:

        result = evaluate_single_query(
            retrieved=case.get("retrieved", []),
            relevant_chunks=case.get(
                "relevant_chunks",
                [],
            ),
            relevant_sources=case.get(
                "relevant_sources",
                [],
            ),
            k=k,
        )

        result["query"] = case["query"]

        results.append(result)

    if not results:
        return {
            "queries": 0,
            "k": k,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "results": [],
        }

    return {
        "queries": len(results),
        "k": k,
        "recall_at_k": sum(
            result["recall_at_k"]
            for result in results
        ) / len(results),
        "precision_at_k": sum(
            result["precision_at_k"]
            for result in results
        ) / len(results),
        "mrr": sum(
            result["reciprocal_rank"]
            for result in results
        ) / len(results),
        "ndcg_at_k": sum(
            result["ndcg_at_k"]
            for result in results
        ) / len(results),
        "results": results,
    }