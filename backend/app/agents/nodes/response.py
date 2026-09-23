from __future__ import annotations

from backend.app.agents.state import NeuroScreenState


DISCLAIMER = (
    "NEUROSCREEN AI is a research/educational screening and support "
    "prototype. It does not provide a medical diagnosis. Screening "
    "results should be interpreted by an appropriately qualified "
    "healthcare professional."
)


def _build_summary(
    state: NeuroScreenState,
) -> str:
    """
    Build a deterministic summary from workflow outputs.

    This function does not perform a new prediction.
    """

    prediction = state.get(
        "prediction",
        {},
    )

    uncertainty = state.get(
        "uncertainty",
        {},
    )

    distribution = state.get(
        "distribution",
        {},
    )

    decision = state.get(
        "decision",
        {},
    )

    route = state.get(
        "route",
        "UNKNOWN",
    )

    predicted_label = prediction.get(
        "label",
        "UNKNOWN",
    )

    probability_yes = prediction.get(
        "probability_yes"
    )

    uncertainty_category = uncertainty.get(
        "category",
        "UNKNOWN",
    )

    distribution_status = distribution.get(
        "status",
        "UNKNOWN",
    )

    decision_status = decision.get(
        "status",
        "UNKNOWN",
    )

    summary_parts = [
        f"Screening result: {predicted_label}."
    ]

    if probability_yes is not None:

        try:

            probability_percent = (
                float(
                    probability_yes
                )
                * 100
            )

            summary_parts.append(
                "Model-estimated positive-class "
                "probability: "
                f"{probability_percent:.4f}%."
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    summary_parts.append(
        "Model uncertainty category: "
        f"{uncertainty_category}."
    )

    summary_parts.append(
        "Distribution status: "
        f"{distribution_status}."
    )

    summary_parts.append(
        "Workflow decision: "
        f"{decision_status}."
    )

    summary_parts.append(
        "Workflow route: "
        f"{route}."
    )

    return " ".join(
        summary_parts
    )


def response_node(
    state: NeuroScreenState,
) -> dict:
    """
    Assemble the final structured response.

    No new prediction is performed here.

    All deterministic ML outputs, RAG evidence, LLM output,
    groundedness, and final audit information are preserved.
    """

    evidence = state.get(
        "retrieved_evidence",
        [],
    )

    citations = state.get(
        "citations",
        [],
    )

    previous_response = dict(
        state.get(
            "response",
            {},
        )
    )

    groundedness = state.get(
        "groundedness",
        previous_response.get(
            "groundedness",
            {},
        ),
    )

    audit = previous_response.get(
        "audit",
        {},
    )

    response = {
        # =========================================================
        # SUMMARY
        # =========================================================

        "summary": _build_summary(
            state
        ),

        # =========================================================
        # ML
        # =========================================================

        "prediction": state.get(
            "prediction",
            {},
        ),

        "model": state.get(
            "model_info",
            {},
        ),

        "uncertainty": state.get(
            "uncertainty",
            {},
        ),

        "distribution": state.get(
            "distribution",
            {},
        ),

        "decision": state.get(
            "decision",
            {},
        ),

        # =========================================================
        # ROUTING
        # =========================================================

        "route": state.get(
            "route",
            "UNKNOWN",
        ),

        "route_reason": state.get(
            "route_reason",
            "",
        ),

        # =========================================================
        # RAG
        # =========================================================

        "evidence": evidence,

        "citations": citations,

        # =========================================================
        # SAFETY
        # =========================================================

        "safety_status": state.get(
            "safety_status",
            "UNKNOWN",
        ),

        "safety_notes": state.get(
            "safety_notes",
            [],
        ),

        # =========================================================
        # LLM
        # =========================================================

        "llm_explanation": previous_response.get(
            "llm_explanation",
            "",
        ),

        "llm_explanation_original": previous_response.get(
            "llm_explanation_original",
            previous_response.get(
                "llm_explanation",
                "",
            ),
        ),

        "generator": previous_response.get(
            "generator",
            "",
        ),

        "fallback_used": previous_response.get(
            "fallback_used",
            False,
        ),

        "final_explanation_status": previous_response.get(
            "final_explanation_status",
            "",
        ),

        # =========================================================
        # GROUNDEDNESS
        # =========================================================

        "groundedness": groundedness,

        "groundedness_passed": groundedness.get(
            "passed",
            False,
        )
        if isinstance(
            groundedness,
            dict,
        )
        else False,

        # =========================================================
        # FINAL AUDIT
        # =========================================================

        "audit": audit,

        # =========================================================
        # DISCLAIMER
        # =========================================================

        "disclaimer": DISCLAIMER,
    }

    return {
        "response": response,
    }