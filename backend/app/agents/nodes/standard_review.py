
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState


def standard_review_node(state: NeuroScreenState) -> dict:
    return {
        "route": "STANDARD_REVIEW",
        "route_reason": (
            "Standard screening workflow selected because "
            "the result is neither highly uncertain nor OOD."
        ),
    }