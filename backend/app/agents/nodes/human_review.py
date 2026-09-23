
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState


def human_review_node(state: NeuroScreenState) -> dict:
    return {
        "route": "HUMAN_REVIEW",
        "route_reason": (
            "Human review branch selected because "
            "the model uncertainty is high."
        ),
    }