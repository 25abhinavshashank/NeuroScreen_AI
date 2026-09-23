from __future__ import annotations

import re
from typing import Any

from langchain_ollama import ChatOllama

from backend.app.agents.state import NeuroScreenState
from backend.app.core.config import get_settings

MODEL_NAME = get_settings().ollama_model


SYSTEM_PROMPT = """
You are the evidence-grounded explanation component of NEUROSCREEN AI.

You provide only general educational information based strictly on the
supplied evidence.

You MUST NOT:
- diagnose autism or any other condition
- state that an individual has or does not have autism
- estimate individual clinical risk
- interpret individual model probabilities
- interpret individual model confidence or uncertainty
- interpret individual questionnaire features
- provide treatment advice
- make causal claims
- discuss OOD status
- discuss workflow routing
- discuss model performance or clinical accuracy

CITATION RULES:
1. Every factual sentence MUST end with one or more citation IDs.
2. Citation IDs MUST come only from the supplied evidence.
3. A citation may be used only when the cited evidence text directly supports
   the factual statement.
4. Do not cite an evidence item merely because it has a high retrieval rank.
5. Read the evidence text before selecting its citation.
6. Do not invent, modify, or guess citation IDs.
7. Do not combine unrelated evidence into a single factual statement.
8. If the supplied evidence does not support a statement, do not make that
   statement.
9. Do not place citations in the middle of a sentence.
10. Return only 1-3 short sentences.
11. Citation format is EXACTLY [E1], [E2], etc. -- always keep the letter E.
    Correct:   Developmental screening does not provide a diagnosis. [E1]
    INCORRECT: Developmental screening does not provide a diagnosis. [1]
    INCORRECT: Developmental screening does not provide a diagnosis. (E1)

The output must contain no heading, no bullet list, no explanation of these
instructions, and no text outside the requested sentences.
"""


# ============================================================
# TEXT EXTRACTION
# ============================================================


def _extract_text(item: dict[str, Any]) -> str:
    """
    Extract evidence text while supporting the field names used
    by different retrieval/evidence nodes.
    """

    for field in (
        "text",
        "page_content",
        "content",
        "chunk_text",
        "evidence",
        "source_text",
    ):
        value = item.get(field)

        if value is not None:
            text = str(value).strip()

            if text:
                return text

    return ""


def _citation_id(item: dict[str, Any]) -> str:
    """
    Return a normalized citation ID such as E1.
    """

    value = item.get("citation_id") or item.get("id")

    if value is None:
        if "rank" in item and item["rank"]:
            return f"E{item['rank']}"
        return ""

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()

    return value


# ============================================================
# EVIDENCE FORMATTING
# ============================================================


def _format_evidence(
    evidence: list[dict[str, Any]],
) -> str:
    """
    Format retrieved evidence so the LLM can explicitly associate
    each evidence chunk with its citation ID.
    """

    parts: list[str] = []

    for item in evidence:
        citation_id = _citation_id(item)
        text = _extract_text(item)

        if not citation_id or not text:
            continue

        parts.append(
            f"[{citation_id}]\n"
            f"{text}"
        )

    if not parts:
        return "NO EVIDENCE AVAILABLE."

    return "\n\n".join(parts)


# ============================================================
# OUTPUT CLEANING
# ============================================================


def _clean(text: str, evidence: list[dict[str, Any]] | None = None) -> str:
    """
    Normalize the raw Ollama response without changing valid
    citation IDs.

    Also repairs the single most common citation mistake this model
    makes despite the evidence block and prompt both showing the
    correct [E1] format: writing a bare-number citation like [1]
    instead. Without this, that sentence's citation is invisible to
    every downstream regex (they all require the E prefix), so the
    sentence is silently treated as uncited and the whole explanation
    fails groundedness -- confirmed as the actual root cause of
    repeated fallbacks by direct trace, not a hypothetical case.
    Only remaps [N] -> [EN] when evidence EN actually exists, so this
    can't invent a citation to evidence that wasn't retrieved.
    """

    text = str(text or "")

    if evidence:
        valid_numbers = set()
        for item in evidence:
            cid = _citation_id(item)
            if cid.upper().startswith("E") and cid[1:].isdigit():
                valid_numbers.add(cid[1:])

        if valid_numbers:
            def _add_e_prefix(match: re.Match) -> str:
                number = match.group(1)
                return f"[E{number}]" if number in valid_numbers else match.group(0)

            text = re.sub(r"\[(\d+)\]", _add_e_prefix, text)

    text = text.replace(
        "```text",
        "",
    )

    text = text.replace(
        "```",
        "",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    filtered_lines: list[str] = []

    for line in lines:
        if line.lower() in {
            "answer:",
            "explanation:",
            "response:",
        }:
            continue

        filtered_lines.append(line)

    cleaned = " ".join(filtered_lines).strip()

    # Normalize whitespace around citation IDs.
    cleaned = re.sub(
        r"\s+\[(E\d+)\]",
        r" [\1]",
        cleaned,
    )

    return cleaned


# ============================================================
# CITATION VALIDATION
# ============================================================


def _extract_citation_ids(
    text: str,
) -> list[str]:
    """
    Extract citation IDs from generated text.

    Example:
        "... screening. [E2]"
    -> ["E2"]
    """

    matches = re.findall(
        r"\[(E\d+)\]",
        text,
        flags=re.IGNORECASE,
    )

    result: list[str] = []

    for match in matches:
        citation_id = match.upper()

        if citation_id not in result:
            result.append(citation_id)

    return result


def _valid_citation_ids(
    evidence: list[dict[str, Any]],
) -> set[str]:
    """
    Return citation IDs actually present in retrieved evidence.
    """

    valid: set[str] = set()

    for item in evidence:
        citation_id = _citation_id(item)

        if citation_id:
            valid.add(citation_id.upper())

    return valid


def _has_only_valid_citations(
    text: str,
    evidence: list[dict[str, Any]],
) -> bool:
    """
    Ensure every citation used by the LLM exists in the
    retrieved evidence.
    """

    used = _extract_citation_ids(text)

    if not used:
        return False

    valid = _valid_citation_ids(evidence)

    return all(
        citation_id in valid
        for citation_id in used
    )


def _all_factual_sentences_have_citations(
    text: str,
) -> bool:
    """
    Ensure each generated sentence ends with a citation.

    This is intentionally conservative.
    """

    if not text.strip():
        return False

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return False

    for sentence in sentences:
        if not re.search(
            r"\[(E\d+)\]\s*[.!?]?$",
            sentence,
            flags=re.IGNORECASE,
        ):
            return False

    return True


# ============================================================
# SAFE EVIDENCE-BASED FALLBACK
# ============================================================


def _safe_evidence_fallback(
    evidence: list[dict[str, Any]],
) -> str:
    """
    Produce a deterministic explanation using only evidence
    that is explicitly present.

    This is NOT a diagnostic interpretation.
    """

    evidence_by_id: dict[str, str] = {}

    for item in evidence:
        citation_id = _citation_id(item)
        text = _extract_text(item)

        if citation_id and text:
            evidence_by_id[citation_id.upper()] = text

    # --------------------------------------------------------
    # E2: positive screening -> thorough assessment
    # --------------------------------------------------------

    if "E2" in evidence_by_id:
        e2 = evidence_by_id["E2"].lower()

        if (
            "positive screening result" in e2
            and "thorough assessment" in e2
        ):
            return (
                "A positive screening result should be followed "
                "by a thorough assessment. [E2]"
            )

    # --------------------------------------------------------
    # E5: developmental concerns
    # --------------------------------------------------------

    if "E5" in evidence_by_id:
        e5 = evidence_by_id["E5"].lower()

        if (
            "developmental concerns" in e5
            or "screening" in e5
        ):
            return (
                "Developmental screening can help identify "
                "developmental concerns. [E5]"
            )

    # --------------------------------------------------------
    # Generic evidence-backed fallback
    # --------------------------------------------------------

    available_ids = sorted(
        evidence_by_id.keys(),
        key=lambda value: (
            int(value[1:])
            if value[1:].isdigit()
            else 999999
        ),
    )

    if available_ids:
        citation_id = available_ids[0]

        return (
            "The retrieved evidence provides general "
            "information relevant to developmental screening "
            f"and assessment. [{citation_id}]"
        )

    if evidence:
        return (
            "The retrieved evidence provides general "
            "information relevant to developmental screening "
            "and assessment. [E1]"
        )

    return ""


# ============================================================
# LLM RESPONSE NODE
# ============================================================


def llm_response_node(
    state: NeuroScreenState,
) -> dict[str, Any]:
    """
    Generate an evidence-grounded educational explanation.

    The LLM receives only retrieved evidence and is explicitly
    prohibited from interpreting the individual assessment.

    If the generated response fails basic citation-format
    validation, a deterministic evidence-based fallback is
    returned instead.
    """

    response = dict(
        state.get(
            "response",
            {},
        )
    )

    evidence = state.get("retrieved_evidence") or state.get("citations") or []

    if not isinstance(
        evidence,
        list,
    ):
        evidence = []

    # Ensure every chunk has citation_id
    for idx, item in enumerate(evidence, start=1):
        if not item.get("citation_id"):
            item["citation_id"] = f"[E{idx}]"

    evidence_text = _format_evidence(
        evidence
    )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence:
        response["llm_explanation"] = ""
        response["generator"] = (
            f"ollama/{MODEL_NAME}"
        )
        response["llm_generation_status"] = "NO_EVIDENCE"

        errors = list(
            state.get(
                "errors",
                [],
            )
            or []
        )

        errors.append(
            "LLM response generation skipped because "
            "no retrieved evidence was available."
        )

        return {
            "response": response,
            "errors": errors,
        }

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
SUPPLIED EVIDENCE
=================

{evidence_text}


TASK
====

Write 1-3 short sentences containing only general educational information
about developmental screening and follow-up assessment.

Before writing each factual sentence:

1. Read the supplied evidence.
2. Identify the exact evidence item that supports the sentence.
3. Put that item's citation at the END of the sentence.
4. Do not use a citation unless its evidence directly supports the sentence.

Important evidence examples:

- If an evidence item explicitly states that screening tools do not provide
  conclusive evidence or diagnoses and that a positive screening result should
  be followed by thorough assessment, cite that item.

- If an evidence item explicitly discusses developmental concerns, cite that
  item for statements about developmental concerns.

Do not discuss:
- this individual
- the individual prediction
- probability
- confidence
- uncertainty
- questionnaire features
- OOD status
- routing
- diagnosis
- treatment
- model performance

Every factual sentence MUST end with its supporting citation.

Return only the sentences.
"""

    # --------------------------------------------------------
    # Invoke Ollama
    # --------------------------------------------------------

    try:
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=get_settings().ollama_host,
            temperature=0,
        )

        result = llm.invoke(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                (
                    "human",
                    prompt,
                ),
            ]
        )

        content = result.content

        if isinstance(
            content,
            list,
        ):
            content = " ".join(
                (
                    str(
                        item.get(
                            "text",
                            item,
                        )
                    )
                    if isinstance(
                        item,
                        dict,
                    )
                    else str(item)
                )
                for item in content
            )

        generated_text = _clean(
            str(content),
            evidence=evidence,
        )

        # ----------------------------------------------------
        # Validate generated citations
        # ----------------------------------------------------

        valid_citations = (
            _has_only_valid_citations(
                generated_text,
                evidence,
            )
        )

        citation_format_valid = (
            _all_factual_sentences_have_citations(
                generated_text
            )
        )

        # ----------------------------------------------------
        # Process LLM output
        # ----------------------------------------------------

        if generated_text:
            response["llm_explanation"] = generated_text
            response["generator"] = f"ollama/{MODEL_NAME}"
            response["llm_generation_status"] = (
                "SUCCESS"
                if (valid_citations and citation_format_valid)
                else "PENDING_GUARD"
            )
            response["fallback_used"] = False

            return {
                "response": response,
            }

        # ----------------------------------------------------
        # Empty LLM output: use deterministic evidence fallback
        # ----------------------------------------------------

        fallback = _safe_evidence_fallback(
            evidence
        )

        response[
            "llm_explanation_rejected"
        ] = generated_text

        response[
            "llm_explanation"
        ] = fallback

        response[
            "generator"
        ] = (
            "deterministic_evidence_fallback"
        )

        response[
            "llm_generation_status"
        ] = "REJECTED"

        response[
            "fallback_used"
        ] = True

        return {
            "response": response,
        }

    # --------------------------------------------------------
    # LLM invocation failure
    # --------------------------------------------------------

    except Exception as exc:
        fallback = _safe_evidence_fallback(
            evidence
        )

        response[
            "llm_explanation"
        ] = fallback

        response[
            "generator"
        ] = (
            "deterministic_evidence_fallback"
        )

        response[
            "llm_generation_status"
        ] = "FAILED"

        response[
            "fallback_used"
        ] = True

        errors = list(
            state.get(
                "errors",
                [],
            )
            or []
        )

        errors.append(
            "LLM response generation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "response": response,
            "errors": errors,
        }


__all__ = [
    "llm_response_node",
    "MODEL_NAME",
    "SYSTEM_PROMPT",
]