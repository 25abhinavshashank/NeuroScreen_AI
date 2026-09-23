
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState


def safety_node(state: NeuroScreenState) -> dict:
    """
    Basic safety gate for the screening workflow.

    This prototype does not make a diagnosis.
    It establishes that the workflow is operating in
    screening/support mode.
    """

    patient_data = state.get("patient_data")

    if not patient_data:
        return {
            "safety_status": "BLOCKED",
            "safety_notes": [
                "Assessment data is missing."
            ],
            "errors": [
                "Safety gate blocked: no assessment data."
            ],
        }

    return {
        "safety_status": "SCREENING_ONLY",
        "safety_notes": [
            "Output is a screening/support result.",
            "The system does not provide a diagnosis.",
            "Professional evaluation is required for diagnostic decisions.",
        ],
    }