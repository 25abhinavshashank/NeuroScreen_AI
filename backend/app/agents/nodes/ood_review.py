
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState


def ood_review_node(state: NeuroScreenState) -> dict:
    return {
        "route": "OOD_REVIEW",
        "route_reason": (
            "OOD review branch selected because "
            "the assessment differs from the configured "
            "training-data distribution."
        ),
    }