
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from backend.app.rag.evaluation.groundedness import (
    DEFAULT_SUPPORT_THRESHOLD,
    assign_citations_to_sentences,
    build_evidence_map,
    evaluate_groundedness,
    lexical_support_score,
    sentence_split,
)


def _get_evidence_items(
    state: Dict[str, Any],
) -> Sequence[Any]:
    """
    Extract evidence from LangGraph state.

    Supports the evidence names used by different stages
    of the NeuroScreen graph.
    """

    for key in (
        "evidence",
        "retrieved_evidence",
        "retrieved_documents",
    ):
        value = state.get(key)

        if isinstance(value, (list, tuple, dict)):
            return value

    return []


def _get_llm_text(
    state: Dict[str, Any],
) -> str:
    """
    Retrieve the generated explanation.

    Canonical layout:
        state["response"]["llm_explanation"]

    Older layouts are also supported.
    """

    response = state.get("response")

    if isinstance(response, dict):
        for key in (
            "llm_explanation",
            "llm_response",
            "generated_response",
            "explanation",
        ):
            value = response.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in (
        "llm_explanation",
        "llm_response",
        "generated_response",
        "explanation",
    ):
        value = state.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _split_sentences_with_citations(
    text: str,
) -> List[Tuple[str, List[str]]]:
    """
    Compatibility wrapper around the canonical groundedness parser.

    Returns:
        [
            ("claim text", ["E1", "E2"]),
            ...
        ]
    """

    sentences = sentence_split(text)

    return assign_citations_to_sentences(sentences)


def _normalize_evidence(
    evidence: Sequence[Any] | Dict[str, Any],
) -> List[dict[str, Any]]:
    """
    Normalize evidence into the list-of-dicts format expected by
    the groundedness evaluator.
    """

    if isinstance(evidence, dict):
        values = list(evidence.values())
    else:
        values = list(evidence or [])

    normalized: List[dict[str, Any]] = []

    for item in values:
        if isinstance(item, dict):
            normalized.append(item)

    return normalized


def _repair_citations(
    response: str,
    evidence: Sequence[Any] | Dict[str, Any],
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> str:
    """
    Repair unsupported citation assignments.

    The factual sentence itself is never rewritten.
    Only citation markers may be changed.
    """

    if not response.strip():
        return response

    normalized_evidence = _normalize_evidence(evidence)

    if not normalized_evidence:
        return response

    evidence_map = build_evidence_map(
        normalized_evidence
    )

    if not evidence_map:
        return response

    parsed_sentences = _split_sentences_with_citations(
        response
    )

    repaired: List[str] = []

    for sentence, original_citations in parsed_sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # --------------------------------------------------------
        # No citation.
        #
        # Do not invent a citation for an uncited claim.
        # --------------------------------------------------------

        if not original_citations:
            repaired.append(sentence)
            continue

        # --------------------------------------------------------
        # Validate existing citations.
        # --------------------------------------------------------

        existing_supported_ids: List[str] = []

        for citation_id in original_citations:

            evidence_text = evidence_map.get(
                citation_id.upper(),
                "",
            )

            if not evidence_text:
                continue

            score = lexical_support_score(
                sentence,
                evidence_text,
            )

            if score >= threshold:
                existing_supported_ids.append(
                    citation_id.upper()
                )

        # --------------------------------------------------------
        # Existing citation is supported.
        # --------------------------------------------------------

        if existing_supported_ids:
            formatted_ids = [
                f"[{cid.strip('[]').upper()}]"
                for cid in existing_supported_ids
            ]
            repaired.append(
                f"{sentence} "
                + " ".join(formatted_ids)
            )
            continue

        # --------------------------------------------------------
        # Existing citation is unsupported.
        #
        # Search only already-retrieved evidence for a
        # sufficiently supporting citation.
        # --------------------------------------------------------

        ranked_evidence: List[
            Tuple[float, str]
        ] = []

        seen_eids = set()
        for evidence_id, evidence_text in evidence_map.items():
            clean_eid = evidence_id.strip("[]").upper()
            if clean_eid in seen_eids:
                continue
            seen_eids.add(clean_eid)

            score = lexical_support_score(
                sentence,
                evidence_text,
            )

            ranked_evidence.append(
                (
                    score,
                    f"[{clean_eid}]",
                )
            )

        ranked_evidence.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        if ranked_evidence:

            best_score, best_id = ranked_evidence[0]

            if best_score >= threshold:

                selected_ids = [best_id]

                for score, evidence_id in ranked_evidence[1:]:

                    if len(selected_ids) >= 2:
                        break

                    if (
                        score >= threshold
                        and score >= best_score * 0.90
                    ):
                        selected_ids.append(
                            evidence_id
                        )

                repaired.append(
                    f"{sentence} "
                    + " ".join(selected_ids)
                )

                continue

        # --------------------------------------------------------
        # No sufficiently supporting evidence found.
        #
        # Preserve the original citation and let groundedness /
        # final audit reject the unsupported claim if necessary.
        # --------------------------------------------------------

        formatted_original = [
            f"[{citation.strip('[]').upper()}]"
            for citation in original_citations
        ]
        repaired.append(
            f"{sentence} "
            + " ".join(formatted_original)
        )

    return " ".join(repaired)


def citation_guard_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    LangGraph citation guard node.

    Responsibilities:
        1. Read the existing LLM explanation.
        2. Read only already-retrieved evidence.
        3. Validate citation/evidence alignment.
        4. Deterministically repair incorrect citations.
        5. Recalculate groundedness.

    It does not:
        - modify ML prediction
        - modify probabilities
        - modify uncertainty
        - modify OOD status
        - diagnose
        - provide treatment
        - retrieve new evidence
        - invent evidence
        - rewrite factual claims
    """

    original_response = _get_llm_text(state)

    evidence = _get_evidence_items(state)

    normalized_evidence = _normalize_evidence(
        evidence
    )

    # --------------------------------------------------------
    # Preserve existing response object.
    # --------------------------------------------------------

    response = state.get("response")

    if isinstance(response, dict):
        updated_response = dict(response)
    else:
        updated_response = {}

    # --------------------------------------------------------
    # No LLM response.
    # --------------------------------------------------------

    if not original_response:

        return {
            "response": updated_response,
            "citation_guard_status": "SKIPPED",
            "citation_guard_changed": False,
            "citation_guard_reason": (
                "No LLM explanation available."
            ),
        }

    # --------------------------------------------------------
    # No evidence.
    # --------------------------------------------------------

    if not normalized_evidence:

        updated_response["llm_explanation"] = (
            original_response
        )

        return {
            "response": updated_response,
            "llm_explanation": original_response,
            "citation_guard_status": "SKIPPED",
            "citation_guard_changed": False,
            "citation_guard_reason": (
                "No retrieved evidence available."
            ),
        }

    # --------------------------------------------------------
    # Repair citations.
    # --------------------------------------------------------

    guarded_response = _repair_citations(
        response=original_response,
        evidence=normalized_evidence,
        threshold=DEFAULT_SUPPORT_THRESHOLD,
    )

    # --------------------------------------------------------
    # Re-evaluate groundedness using the REAL evaluator API.
    # --------------------------------------------------------

    groundedness_result = evaluate_groundedness(
        answer=guarded_response,
        citations=normalized_evidence,
        evidence=None,
        threshold=DEFAULT_SUPPORT_THRESHOLD,
    )

    changed = (
        guarded_response != original_response
    )

    # --------------------------------------------------------
    # Preserve canonical response structure.
    # --------------------------------------------------------

    updated_response["llm_explanation"] = (
        guarded_response
    )

    return {
        "response": updated_response,

        # Backward-compatible top-level field.
        "llm_explanation": guarded_response,

        "citation_guard_status": (
            "APPLIED"
            if changed
            else "VALIDATED"
        ),

        "citation_guard_changed": changed,

        "citation_guard_original": (
            original_response
            if changed
            else None
        ),

        "citation_guard_groundedness": (
            groundedness_result
        ),

        # Downstream compatibility.
        "groundedness": groundedness_result,
    }


__all__ = [
    "citation_guard_node",
]