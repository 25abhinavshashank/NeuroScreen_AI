from __future__ import annotations

from typing import Any, List


def _extract_text(document: dict[str, Any]) -> str:
    """
    Extract evidence text from a retrieved document.

    Different retrieval/loader implementations may use different
    field names, so we support the common variants.
    """

    possible_fields = (
        "text",
        "page_content",
        "content",
        "chunk_text",
    )

    for field in possible_fields:
        value = document.get(field)

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


def build_citations(
    retrieved_documents: List[dict],
) -> List[dict]:
    """
    Build citation metadata while preserving the actual
    retrieved evidence text.

    Citation contract:

    {
        "citation_id": "[E1]",
        "source": "...",
        "file_name": "...",
        "chunk_id": "...",
        "similarity": 0.85,
        "text": "actual evidence text"
    }

    The evidence text is intentionally preserved so downstream
    components such as the groundedness evaluator can verify
    whether generated claims are supported.
    """

    citations: List[dict[str, Any]] = []

    for rank, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        text = _extract_text(document)

        citations.append(
            {
                "citation_id": f"[E{rank}]",
                "source": document.get("source"),
                "file_name": document.get("file_name"),
                "chunk_id": document.get("chunk_id"),
                "similarity": document.get("similarity"),
                "text": text,
            }
        )

    return citations