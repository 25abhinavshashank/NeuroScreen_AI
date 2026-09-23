
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState


def routing_node(state: NeuroScreenState) -> dict:
    """
    Decide which workflow branch should handle the assessment.

    Routing priority:
    1. OOD -> OOD_REVIEW
    2. HIGH uncertainty -> HUMAN_REVIEW
    3. Otherwise -> STANDARD_REVIEW
    """

    decision = state.get("decision", {})
    distribution = state.get("distribution", {})
    uncertainty = state.get("uncertainty", {})

    if not decision:
        return {
            "route": "ERROR",
            "route_reason": "Decision information is missing.",
            "errors": ["Routing node could not find decision information."],
        }

    # OOD takes highest priority.
    if distribution.get("is_ood") is True:
        return {
            "route": "OOD_REVIEW",
            "route_reason": (
                "The assessment is outside the configured "
                "training-data distribution."
            ),
        }

    # High uncertainty requires human review.
    if uncertainty.get("category") == "HIGH":
        return {
            "route": "HUMAN_REVIEW",
            "route_reason": (
                "The model reports high uncertainty, "
                "so the result should receive human review."
            ),
        }

    # Normal screening workflow.
    return {
        "route": "STANDARD_REVIEW",
        "route_reason": (
            "The assessment passed the configured "
            "uncertainty and distribution checks."
        ),
    }


def route_after_classifier(state: NeuroScreenState) -> str:
    """
    LangGraph conditional-edge function.
    """

    route = state.get("route", "ERROR")

    if route == "OOD_REVIEW":
        return "ood_review"

    if route == "HUMAN_REVIEW":
        return "human_review"

    if route == "STANDARD_REVIEW":
        return "standard_review"

    return "error"