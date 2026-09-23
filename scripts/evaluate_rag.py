from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.app.rag.retrieval.vector_store import VectorStore
from backend.app.rag.reranking.fusion import rerank_candidates
from backend.app.rag.evaluation.retrieval import (
    evaluate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "evaluation_dataset.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)

REPORT_PATH = (
    REPORT_DIR
    / "rag_retrieval_report.json"
)


TOP_K = 5
CANDIDATE_K = 20


def load_dataset() -> list[dict[str, Any]]:
    """Load evaluation queries."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Evaluation dataset must be a JSON list."
        )

    return dataset


def retrieve_evidence(
    dataset: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run every evaluation query through the real FAISS VectorStore,
    over-fetching a candidate pool and then applying the same hybrid
    (dense + BM25) reranker used in the live LangGraph pipeline.

    Returns (raw_evaluations, reranked_evaluations) so the report can
    show the reranker's actual before/after effect on retrieval quality.
    """

    vector_store = VectorStore()

    print()
    print("Loading FAISS vector store...")
    vector_store.load()

    print()

    raw_evaluations = []
    reranked_evaluations = []

    for number, case in enumerate(
        dataset,
        start=1,
    ):

        query = case["query"]

        print(
            f"[{number}/{len(dataset)}] "
            f"Query: {query}"
        )

        candidates = vector_store.search(
            query=query,
            top_k=CANDIDATE_K,
        )

        retrieved = candidates[:TOP_K]
        reranked = rerank_candidates(query, candidates, top_n=TOP_K)

        print(
            f"    Retrieved: "
            f"{len(candidates)} candidates, "
            f"reranked to {len(reranked)}"
        )

        for rank, item in enumerate(
            retrieved,
            start=1,
        ):
            chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("metadata", {}).get("chunk_id")
                or item.get("metadata", {}).get("id")
            )

            source = (
                item.get("source")
                or item.get("source_name")
                or item.get("metadata", {}).get("source")
                or item.get("metadata", {}).get("source_name")
                or item.get("metadata", {}).get("file_name")
            )

            similarity = item.get(
                "similarity",
                0.0,
            )

            print(
                f"       {rank}. "
                f"{chunk_id} | "
                f"{source} | "
                f"{similarity:.6f}"
            )

        raw_evaluations.append(
            {
                **case,
                "retrieved": retrieved,
            }
        )
        reranked_evaluations.append(
            {
                **case,
                "retrieved": reranked,
            }
        )

    return raw_evaluations, reranked_evaluations


def save_report(
    report: dict[str, Any],
) -> None:
    """Save evaluation report."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    label: str,
    report: dict[str, Any],
) -> None:
    """Print a summary section for one report (raw or reranked)."""

    print()
    print("-" * 60)
    print(label)
    print("-" * 60)

    print(f"Queries evaluated : {report['queries']}")
    print(f"K                 : {report['k']}")
    print()
    print(f"Recall@{report['k']}    : {report['recall_at_k']:.4f}")
    print(f"Precision@{report['k']} : {report['precision_at_k']:.4f}")
    print(f"MRR               : {report['mrr']:.4f}")
    print(f"NDCG@{report['k']}      : {report['ndcg_at_k']:.4f}")

    for number, result in enumerate(report["results"], start=1):
        print()
        print(f"{number}. {result['query']}")
        print(f"   Retrieved chunks : {result['retrieved_chunks']}")
        print(f"   Chunk hits       : {result['chunk_hits']}")
        print(f"   Recall@{report['k']}       : {result['recall_at_k']:.4f}")
        print(f"   Precision@{report['k']}    : {result['precision_at_k']:.4f}")
        print(f"   Reciprocal Rank  : {result['reciprocal_rank']:.4f}")
        print(f"   NDCG@{report['k']}         : {result['ndcg_at_k']:.4f}")


def main() -> None:

    dataset = load_dataset()

    raw_evaluations, reranked_evaluations = retrieve_evidence(
        dataset
    )

    raw_report = evaluate_dataset(raw_evaluations, k=TOP_K)
    reranked_report = evaluate_dataset(reranked_evaluations, k=TOP_K)

    combined_report = {
        "raw_faiss": raw_report,
        "reranked_hybrid_rrf": reranked_report,
    }

    save_report(combined_report)

    print()
    print("=" * 60)
    print("NEUROSCREEN AI — RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    print_summary("RAW FAISS (dense similarity only)", raw_report)
    print_summary("RERANKED (hybrid BM25 + dense, RRF fusion)", reranked_report)

    print()
    print("-" * 60)
    print(f"Report saved to: {REPORT_PATH}")
    print("-" * 60)


if __name__ == "__main__":
    main()