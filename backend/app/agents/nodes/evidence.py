from __future__ import annotations

from typing import Any

from backend.app.agents.state import NeuroScreenState


def _extract_text(item: dict[str, Any]) -> str:
    """
    Extract usable evidence text from a retrieval item.
    """

    possible_fields = (
        "text",
        "page_content",
        "content",
        "chunk_text",
    )

    for field in possible_fields:
        value = item.get(field)

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


def evidence_node(state: NeuroScreenState) -> dict:
    """
    Validate and organize retrieved evidence before it reaches
    the LLM and groundedness evaluator.

    The node ensures that every valid evidence item contains
    a normalized `text` field.
    """

    evidence = state.get(
        "retrieved_evidence",
        [],
    )

    citations = state.get(
        "citations",
        [],
    )

    valid_evidence: list[dict[str, Any]] = []

    for index, item in enumerate(evidence, start=1):
        normalized_item = dict(item)
        text = _extract_text(normalized_item)

        if not text:
            continue

        normalized_item["text"] = text

        # Ensure citation_id is always present on retrieved_evidence items
        if not normalized_item.get("citation_id"):
            normalized_item["citation_id"] = f"[E{len(valid_evidence) + 1}]"

        valid_evidence.append(normalized_item)

    # ------------------------------------------------------------
    # Synchronize citation evidence with retrieved evidence 1-to-1.
    # ------------------------------------------------------------
    normalized_citations: list[dict[str, Any]] = []

    for index, item in enumerate(valid_evidence, start=1):
        cid = item.get("citation_id") or f"[E{index}]"
        normalized_citations.append(
            {
                "citation_id": cid,
                "source": item.get("source"),
                "file_name": item.get("file_name"),
                "chunk_id": item.get("chunk_id"),
                "similarity": item.get("similarity"),
                "text": item.get("text", ""),
            }
        )

    return {
        "retrieved_evidence": valid_evidence,
        "citations": normalized_citations,
    }