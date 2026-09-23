from __future__ import annotations

import re
from typing import Any

from backend.app.agents.state import NeuroScreenState


# ============================================================
# CONFIGURATION
# ============================================================

GROUNDEDNESS_THRESHOLD = 0.70
MAX_UNSUPPORTED_CLAIM_RATE = 0.30


# ============================================================
# FORBIDDEN PATTERNS
# ============================================================

FORBIDDEN_DIAGNOSTIC_PATTERNS = [
    r"\bdiagnosed with autism\b",
    r"\bdiagnosed with asd\b",
    r"\bhas autism\b",
    r"\bhas asd\b",
    r"\bhas autism spectrum disorder\b",
    r"\bthe child has autism\b",
    r"\bthe child has asd\b",
    r"\bthe child is autistic\b",
    r"\bthis child is autistic\b",
    r"\bthe patient has autism\b",
    r"\bthe patient has asd\b",
    r"\bthis patient has autism\b",
]

FORBIDDEN_PROBABILITY_PATTERNS = [
    r"\bprobability of autism\b",
    r"\blikelihood of autism\b",
    r"\bchance of autism\b",
    r"\bprobability that the child has autism\b",
    r"\blikely has autism\b",
    r"\bunlikely to have autism\b",
]

FORBIDDEN_RELIABILITY_PATTERNS = [
    r"\bclinically accurate\b",
    r"\bclinical accuracy\b",
    r"\bclinically validated\b",
    r"\bclinical validation\b",
    r"\bclinically reliable\b",
    r"\bclinical reliability\b",
    r"\bmedically accurate\b",
    r"\bmedical accuracy\b",
    r"\bdiagnostically accurate\b",
    r"\bdiagnostic accuracy\b",
    r"\bproven accurate\b",
    r"\bhighly accurate\b",
]

FORBIDDEN_CAUSAL_PATTERNS = [
    r"\bcauses autism\b",
    r"\bcaused autism\b",
    r"\bleads to autism\b",
    r"\bresults in autism\b",
    r"\bcauses asd\b",
    r"\bcaused asd\b",
]

FORBIDDEN_FEATURE_INTERPRETATION_PATTERNS = [
    r"\bthis feature proves\b",
    r"\bthis feature indicates autism\b",
    r"\bthis answer proves autism\b",
    r"\bthis response proves autism\b",
    r"\bthis score proves autism\b",
    r"\bthis behavior proves autism\b",
]

FORBIDDEN_INDIVIDUAL_INTERPRETATION_PATTERNS = [
    r"\bthis child is autistic\b",
    r"\bthe child is autistic\b",
    r"\bthis patient is autistic\b",
    r"\bthe patient is autistic\b",
    r"\bthe child definitely has\b",
    r"\bthe child certainly has\b",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _contains_pattern(
    text: str,
    patterns: list[str],
) -> tuple[bool, str | None]:
    """
    Check whether text contains any forbidden pattern.

    Returns:
        (True, matched_pattern)
        or
        (False, None)
    """

    for pattern in patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return True, pattern

    return False, None


def _as_text(value: Any) -> str:
    """
    Safely convert a possible response value into text.

    None, empty strings, and non-string values that do not
    contain useful text become an empty string.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _get_llm_text(
    state: NeuroScreenState,
    response: dict[str, Any],
) -> str:
    """
    Retrieve the generated explanation from supported response
    locations.

    Primary location:
        response["llm_explanation"]

    Compatibility fallbacks:
        response["llm_response"]
        response["generated_response"]
        response["explanation"]
        state["llm_explanation"]
        state["llm_response"]
        state["generated_response"]
    """

    response_candidates = [
        response.get("llm_explanation"),
        response.get("llm_response"),
        response.get("generated_response"),
        response.get("explanation"),
    ]

    for candidate in response_candidates:
        text = _as_text(candidate)

        if text:
            return text

    state_candidates = [
        state.get("llm_explanation"),
        state.get("llm_response"),
        state.get("generated_response"),
    ]

    for candidate in state_candidates:
        text = _as_text(candidate)

        if text:
            return text

    return ""


def _validate_citations(
    text: str,
    citations: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """
    Validate citation IDs used by the generated explanation.
    """

    used_citations = re.findall(
        r"\[E\d+\]",
        text,
    )

    if not used_citations:
        return True, None

    available_citations = {
        f"[{str(item.get('citation_id')).strip('[]')}]"
        for item in citations
        if item.get("citation_id")
    }

    for citation_id in used_citations:
        if citation_id not in available_citations:
            return False, citation_id

    return True, None


def _fallback_citation_text(
    citation_id: str,
) -> str:
    """
    Normalize a citation ID to the project citation format.
    """

    normalized = str(citation_id).strip()

    if not normalized:
        return ""

    if normalized.startswith("[") and normalized.endswith("]"):
        return normalized

    return f"[{normalized}]"


def _deterministic_safe_response(
    state: NeuroScreenState,
) -> str:
    """
    Deterministic fallback response.

    This response deliberately does not:
    - diagnose
    - interpret probability clinically
    - interpret individual features
    - claim clinical validity
    - make causal claims
    """

    citations = state.get(
        "citations",
        [],
    )

    if not isinstance(
        citations,
        list,
    ):
        citations = []

    citation_ids = [
        str(item.get("citation_id")).strip()
        for item in citations
        if item.get("citation_id")
    ]

    if citation_ids:
        normalized_ids = [
            _fallback_citation_text(cid)
            for cid in citation_ids[:3]
        ]

        citation_text = ", ".join(normalized_ids)

        return (
            "The retrieved evidence provides general "
            "information about developmental screening, "
            "monitoring, and follow-up assessment. "
            f"{citation_text}"
        )

    return (
        "The system retrieved general information about "
        "developmental screening and follow-up assessment."
    )


# ============================================================
# GROUNDEDNESS HELPERS
# ============================================================

def _get_groundedness(
    state: NeuroScreenState,
) -> dict[str, Any]:
    """
    Retrieve groundedness information produced by the
    groundedness node.

    Groundedness is stored at the top-level graph state.
    """

    groundedness = state.get(
        "groundedness",
        {},
    )

    if not isinstance(
        groundedness,
        dict,
    ):
        return {}

    return groundedness


def _groundedness_values(
    groundedness: dict[str, Any],
) -> tuple[float, float]:
    """
    Extract groundedness score and unsupported claim rate.

    Fail closed if values are missing or invalid.
    """

    try:
        groundedness_score = float(
            groundedness.get(
                "groundedness_score",
                groundedness.get(
                    "score",
                    0.0,
                ),
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        groundedness_score = 0.0

    try:
        unsupported_value = groundedness.get(
            "unsupported_claim_rate",
            1.0,
        )

        if unsupported_value is None:
            unsupported_claim_rate = 1.0
        else:
            unsupported_claim_rate = float(
                unsupported_value
            )

    except (
        TypeError,
        ValueError,
    ):
        unsupported_claim_rate = 1.0

    return (
        groundedness_score,
        unsupported_claim_rate,
    )


# ============================================================
# MAIN AUDIT NODE
# ============================================================

def final_audit_node(
    state: NeuroScreenState,
) -> NeuroScreenState:
    """
    Deterministic final safety + groundedness audit.

    The LLM is NOT trusted to determine whether its own
    output is safe or evidence-grounded.

    If an audit check fails, the generated response is
    rejected and replaced with a deterministic safe fallback.
    """

    response = state.get(
        "response",
        {},
    )

    if not isinstance(
        response,
        dict,
    ):
        response = {}

    # --------------------------------------------------------
    # Retrieve generated explanation robustly.
    # --------------------------------------------------------

    llm_text = _get_llm_text(
        state,
        response,
    )

    citations = state.get(
        "citations",
        [],
    )

    if not isinstance(
        citations,
        list,
    ):
        citations = []

    # --------------------------------------------------------
    # Groundedness
    # --------------------------------------------------------

    groundedness = _get_groundedness(
        state
    )

    (
        groundedness_score,
        unsupported_claim_rate,
    ) = _groundedness_values(
        groundedness
    )

    # --------------------------------------------------------
    # Initial status
    # --------------------------------------------------------

    llm_status = "PASSED"
    failure_reason: str | None = None
    matched_pattern: str | None = None

    checks: dict[str, bool] = {
        "non_empty": bool(llm_text),
        "diagnostic_claims": True,
        "probability_claims": True,
        "reliability_claims": True,
        "causal_claims": True,
        "feature_claims": True,
        "individual_interpretation": True,
        "citations": True,
        "groundedness": True,
        "unsupported_claims": True,
    }

    # ========================================================
    # CHECK 1 — NON-EMPTY RESPONSE
    # ========================================================

    if not llm_text:
        checks["non_empty"] = False

        llm_status = "FAILED"

        failure_reason = (
            "LLM generated an empty explanation."
        )

    # ========================================================
    # CHECK 2 — DIAGNOSTIC CLAIMS
    # ========================================================

    if failure_reason is None:
        found, pattern = _contains_pattern(
            llm_text,
            FORBIDDEN_DIAGNOSTIC_PATTERNS,
        )

        if found:
            checks["diagnostic_claims"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM generated a diagnostic claim."
            )

            matched_pattern = pattern

    # ========================================================
    # CHECK 3 — PROBABILITY CLAIMS
    # ========================================================

    if failure_reason is None:
        found, pattern = _contains_pattern(
            llm_text,
            FORBIDDEN_PROBABILITY_PATTERNS,
        )

        if found:
            checks["probability_claims"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM generated a clinical probability "
                "interpretation."
            )

            matched_pattern = pattern

    # ========================================================
    # CHECK 4 — RELIABILITY CLAIMS
    # ========================================================

    if failure_reason is None:
        found, pattern = _contains_pattern(
            llm_text,
            FORBIDDEN_RELIABILITY_PATTERNS,
        )

        if found:
            checks["reliability_claims"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM made an unsupported reliability or "
                "clinical validation claim."
            )

            matched_pattern = pattern

    # ========================================================
    # CHECK 5 — CAUSAL CLAIMS
    # ========================================================

    if failure_reason is None:
        found, pattern = _contains_pattern(
            llm_text,
            FORBIDDEN_CAUSAL_PATTERNS,
        )

        if found:
            checks["causal_claims"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM generated a causal claim."
            )

            matched_pattern = pattern

    # ========================================================
    # CHECK 6 — FEATURE INTERPRETATION
    # ========================================================

    if failure_reason is None:
        found, pattern = _contains_pattern(
            llm_text,
            FORBIDDEN_FEATURE_INTERPRETATION_PATTERNS,
        )

        if found:
            checks["feature_claims"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM generated an unsupported feature "
                "interpretation."
            )

            matched_pattern = pattern

    # ========================================================
    # CHECK 7 — INDIVIDUAL INTERPRETATION
    # ========================================================

    if failure_reason is None:
        found, pattern = _contains_pattern(
            llm_text,
            FORBIDDEN_INDIVIDUAL_INTERPRETATION_PATTERNS,
        )

        if found:
            checks["individual_interpretation"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM generated an unsupported "
                "individual-level interpretation."
            )

            matched_pattern = pattern

    # ========================================================
    # CHECK 8 — CITATION VALIDITY
    # ========================================================

    if failure_reason is None:
        citations_valid, invalid_citation = (
            _validate_citations(
                llm_text,
                citations,
            )
        )

        if not citations_valid:
            checks["citations"] = False
            llm_status = "FAILED"

            failure_reason = (
                "LLM used a citation ID that does not "
                "exist in the retrieved evidence."
            )

            matched_pattern = invalid_citation

    # ========================================================
    # CHECK 9 — GROUNDEDNESS
    # ========================================================

    if failure_reason is None:
        if groundedness_score < GROUNDEDNESS_THRESHOLD:
            checks["groundedness"] = False
            llm_status = "FAILED"

            failure_reason = (
                "Generated explanation did not meet the "
                "configured evidence-groundedness threshold."
            )

            matched_pattern = (
                "groundedness_score="
                f"{groundedness_score:.4f}"
            )

    # ========================================================
    # CHECK 10 — UNSUPPORTED CLAIM RATE
    # ========================================================

    if failure_reason is None:
        if unsupported_claim_rate > MAX_UNSUPPORTED_CLAIM_RATE:
            checks["unsupported_claims"] = False
            llm_status = "FAILED"

            failure_reason = (
                "Generated explanation contains too many "
                "unsupported cited claims."
            )

            matched_pattern = (
                "unsupported_claim_rate="
                f"{unsupported_claim_rate:.4f}"
            )

    # ========================================================
    # BUILD AUDIT RESULT
    # ========================================================

    audit = {
        "llm_status": llm_status,

        "final_status": (
            "SAFE_FALLBACK"
            if llm_status == "FAILED"
            else "PASSED"
        ),

        "fallback_used": (
            llm_status == "FAILED"
        ),

        "reason": (
            failure_reason
            if failure_reason
            else (
                "All deterministic safety and "
                "groundedness checks passed."
            )
        ),

        "checks": checks,

        "matched_pattern": matched_pattern,

        "groundedness": {
            "score": groundedness_score,

            "citation_coverage": groundedness.get(
                "citation_coverage",
                0.0,
            ),

            "citation_precision": groundedness.get(
                "citation_precision",
                0.0,
            ),

            "unsupported_claim_rate": (
                unsupported_claim_rate
            ),

            "total_claims": groundedness.get(
                "total_claims",
                0,
            ),

            "supported_claims": groundedness.get(
                "supported_claims",
                0,
            ),

            "unsupported_claims": groundedness.get(
                "unsupported_claims",
                0,
            ),
        },
    }

    # ========================================================
    # IF AUDIT FAILED
    # ========================================================

    if llm_status == "FAILED":

        # Preserve the rejected output exactly.
        response[
            "llm_explanation_rejected"
        ] = llm_text

        # Deterministic safe fallback.
        response[
            "llm_explanation"
        ] = _deterministic_safe_response(
            state
        )

        response[
            "generator"
        ] = "deterministic_safe_fallback"

        response[
            "fallback_used"
        ] = True

        errors = state.setdefault(
            "errors",
            [],
        )

        if not isinstance(
            errors,
            list,
        ):
            errors = []
            state["errors"] = errors

        errors.append(
            "Final audit rejected the LLM explanation: "
            + str(failure_reason)
        )

    # ========================================================
    # IF AUDIT PASSED
    # ========================================================

    else:
        # Only default to False: groundedness_node may have already
        # substituted a deterministic evidence-grounded fallback that
        # itself passes every audit check. That's still a fallback, and
        # must not be reported as the model's own explanation.
        response.setdefault(
            "fallback_used",
            False,
        )

        response.setdefault(
            "generator",
            "ollama/gemma2:2b",
        )

    # ========================================================
    # STORE AUDIT
    # ========================================================

    response[
        "audit"
    ] = audit

    state[
        "response"
    ] = response

    return state


__all__ = [
    "final_audit_node",
    "GROUNDEDNESS_THRESHOLD",
    "MAX_UNSUPPORTED_CLAIM_RATE",
]