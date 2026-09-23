from __future__ import annotations

import re
from typing import Any


DEFAULT_SUPPORT_THRESHOLD = 0.35
CLAIM_SUPPORT_THRESHOLD = DEFAULT_SUPPORT_THRESHOLD


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "could", "for", "from", "has", "have", "having",
    "how", "if", "in", "into", "is", "it", "its", "may", "might",
    "of", "on", "or", "our", "should", "that", "the", "their",
    "them", "these", "this", "those", "to", "was", "were", "what",
    "when", "which", "who", "with", "would", "you", "your",
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()

    text = re.sub(
        r"\[e\d+\]",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)

    if not normalized:
        return set()

    tokens = normalized.split()

    return {
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 2
    }


def lexical_support_score(
    claim: str,
    evidence: str,
) -> float:
    """
    Deterministic lexical-support heuristic.

    This does NOT establish:
    - semantic entailment
    - factual correctness
    - medical validity
    - clinical correctness
    - causality
    """

    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)

    if not claim_tokens:
        return 0.0

    if not evidence_tokens:
        return 0.0

    overlap = claim_tokens & evidence_tokens

    score = len(overlap) / len(claim_tokens)

    return round(score, 4)


# Backward-compatible alias.
# citation_guard.py imports this name.
support_score = lexical_support_score


def sentence_split(text: str) -> list[str]:
    """
    Split text into sentences while preserving citation markers.

    Examples:

        "Screening does not provide a diagnosis. [E2]"

    becomes:

        [
            "Screening does not provide a diagnosis. [E2]"
        ]

    Also handles:

        "Screening does not provide a diagnosis [E2]."

    and:

        "Screening does not provide a diagnosis. [E2], [E5]."
    """

    text = str(text or "").strip()

    if not text:
        return []

    # First split on normal sentence boundaries.
    raw_parts = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    sentences: list[str] = []

    pending_citations: list[str] = []

    for part in raw_parts:
        part = part.strip()

        if not part:
            continue

        # A citation-only fragment belongs to the previous sentence.
        citation_only = re.fullmatch(
            r"(?:\[E\d+\][,\s]*)+",
            part,
            flags=re.IGNORECASE,
        )

        if citation_only:
            if sentences:
                sentences[-1] = (
                    sentences[-1].rstrip()
                    + " "
                    + part
                )

            else:
                pending_citations.append(part)

            continue

        # Check if part begins with leading citations from previous sentence
        leading_citations_match = re.match(
            r"^((?:\[E\d+\][,\s]*)+)\s*(.*)$",
            part,
            flags=re.IGNORECASE,
        )

        if leading_citations_match:
            leading_cits = leading_citations_match.group(1).strip()
            rest = leading_citations_match.group(2).strip()

            if sentences:
                sentences[-1] = (
                    sentences[-1].rstrip()
                    + " "
                    + leading_cits
                )
            elif leading_cits:
                pending_citations.append(leading_cits)

            part = rest

            if not part:
                continue

        if pending_citations:
            part = (
                part
                + " "
                + " ".join(pending_citations)
            )
            pending_citations.clear()

        sentences.append(part)

    return sentences


def extract_citation_ids(
    text: str,
) -> list[str]:
    """
    Extract citation IDs such as [E1], [E2], [E5].

    Preserves order and removes duplicates.
    """

    matches = re.findall(
        r"\[E\d+\]",
        str(text or ""),
        flags=re.IGNORECASE,
    )

    result: list[str] = []

    for match in matches:
        citation_id = match.upper()

        if citation_id not in result:
            result.append(citation_id)

    return result


def assign_citations_to_sentences(
    sentences: list[str],
) -> list[tuple[str, list[str]]]:
    """
    Associate citations with each sentence.

    Citation markers are removed from the claim text but
    preserved in the citation ID list.
    """

    results: list[tuple[str, list[str]]] = []

    for sentence in sentences:

        citation_ids = extract_citation_ids(
            sentence
        )

        clean_sentence = re.sub(
            r"\[E\d+\]",
            "",
            sentence,
            flags=re.IGNORECASE,
        )

        clean_sentence = re.sub(
            r"\s+",
            " ",
            clean_sentence,
        ).strip()

        # Remove dangling punctuation caused by removing citations.
        clean_sentence = re.sub(
            r"\s+([,.!?])",
            r"\1",
            clean_sentence,
        )

        results.append(
            (
                clean_sentence,
                citation_ids,
            )
        )

    return results


def build_evidence_map(
    citations: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Build:

        [E1] -> evidence text
        E1   -> evidence text
        ...
    """

    evidence_map: dict[str, str] = {}

    for item in citations or []:

        if not isinstance(item, dict):
            continue

        raw_id = (
            item.get("citation_id")
            or item.get("id")
        )

        text = item.get(
            "text",
            "",
        )

        if not raw_id:
            continue

        if not text:
            continue

        cid = str(raw_id).strip().upper()
        evidence_map[cid] = str(text)

        clean_cid = cid.strip("[]")
        evidence_map[clean_cid] = str(text)
        evidence_map[f"[{clean_cid}]"] = str(text)

    return evidence_map


class ClaimEvaluation:
    def __init__(
        self,
        claim: str,
        citation_ids: list[str],
        supported: bool,
        support_score: float,
        matched_evidence: list[str],
    ) -> None:

        self.claim = claim
        self.citation_ids = citation_ids
        self.supported = supported
        self.support_score = support_score
        self.matched_evidence = matched_evidence


def evaluate_claim(
    claim: str,
    citation_ids: list[str],
    evidence_map: dict[str, str],
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> ClaimEvaluation:
    """
    Evaluate whether at least one cited evidence chunk
    provides sufficient lexical support for the claim.
    """

    matched_evidence: list[str] = []

    best_score = 0.0

    for citation_id in citation_ids:

        cid = str(citation_id).strip().upper()
        clean_id = cid.strip("[]")

        evidence = (
            evidence_map.get(cid)
            or evidence_map.get(clean_id)
            or evidence_map.get(f"[{clean_id}]")
            or ""
        )

        if not evidence:
            continue

        score = lexical_support_score(
            claim,
            evidence,
        )

        best_score = max(
            best_score,
            score,
        )

        if score >= threshold:
            matched_evidence.append(
                citation_id
            )

    supported = (
        len(matched_evidence) > 0
    )

    return ClaimEvaluation(
        claim=claim,
        citation_ids=citation_ids,
        supported=supported,
        support_score=round(
            best_score,
            4,
        ),
        matched_evidence=matched_evidence,
    )


def evaluate_groundedness(
    answer: str,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None = None,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """
    Evaluate citation groundedness.

    This is a deterministic lexical-support heuristic.

    It does NOT prove:
    - factual truth
    - semantic entailment
    - medical validity
    - clinical correctness
    - causality
    """

    if isinstance(
        threshold,
        bool,
    ) or not isinstance(
        threshold,
        (int, float),
    ):
        raise TypeError(
            "groundedness threshold must be a numeric float; "
            f"received {type(threshold).__name__}"
        )

    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "groundedness threshold must be between 0.0 and 1.0"
        )

    answer = str(
        answer or ""
    ).strip()

    citations = citations or []

    if not answer:
        return {
            "groundedness_score": 0.0,
            "citation_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
            "citation_precision": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "claims": [],
        }

    evidence_map = build_evidence_map(
        citations
    )

    # Merge separately supplied evidence.
    for item in evidence or []:

        if not isinstance(item, dict):
            continue

        raw_id = (
            item.get("citation_id")
            or item.get("id")
        )

        text = item.get(
            "text",
            "",
        )

        if raw_id and text:
            cid = str(raw_id).strip().upper()
            evidence_map[cid] = str(text)
            clean_cid = cid.strip("[]")
            evidence_map[clean_cid] = str(text)
            evidence_map[f"[{clean_cid}]"] = str(text)

    sentences = sentence_split(
        answer
    )

    sentence_claims = assign_citations_to_sentences(
        sentences
    )

    claims: list[ClaimEvaluation] = []

    cited_sentence_count = 0

    for claim, citation_ids in sentence_claims:

        if not claim:
            continue

        if citation_ids:

            cited_sentence_count += 1

            evaluation = evaluate_claim(
                claim=claim,
                citation_ids=citation_ids,
                evidence_map=evidence_map,
                threshold=threshold,
            )

            claims.append(
                evaluation
            )

        else:
            # An uncited factual sentence counts as an unsupported claim
            claims.append(
                ClaimEvaluation(
                    claim=claim,
                    citation_ids=[],
                    supported=False,
                    support_score=0.0,
                    matched_evidence=[],
                )
            )

    if not claims:
        return {
            "groundedness_score": 0.0,
            "citation_coverage": 0.0,
            "unsupported_claim_rate": 0.0,
            "citation_precision": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "claims": [],
        }

    supported_count = sum(
        1
        for claim in claims
        if claim.supported
    )

    unsupported_count = (
        len(claims)
        - supported_count
    )

    groundedness_score = (
        supported_count
        / len(claims)
    )

    citation_coverage = (
        cited_sentence_count
        / len(sentences)
        if sentences
        else 0.0
    )

    unsupported_claim_rate = (
        unsupported_count
        / len(claims)
    )

    citation_precision = (
        supported_count
        / len(claims)
    )

    return {
        "groundedness_score": round(
            groundedness_score,
            4,
        ),
        "citation_coverage": round(
            citation_coverage,
            4,
        ),
        "unsupported_claim_rate": round(
            unsupported_claim_rate,
            4,
        ),
        "citation_precision": round(
            citation_precision,
            4,
        ),
        "total_claims": len(
            claims
        ),
        "supported_claims": supported_count,
        "unsupported_claims": unsupported_count,
        "claims": [
            {
                "claim": claim.claim,
                "citation_ids": claim.citation_ids,
                "supported": claim.supported,
                "support_score": claim.support_score,
                "matched_evidence": claim.matched_evidence,
            }
            for claim in claims
        ],
    }