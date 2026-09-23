from __future__ import annotations

from typing import Any, TypedDict


class NeuroScreenState(TypedDict, total=False):
    patient_data: dict[str, Any]
    query: str
    safety_status: str
    safety_notes: list[str]
    prediction: dict[str, Any]
    model_info: dict[str, Any]
    uncertainty: dict[str, Any]
    distribution: dict[str, Any]
    decision: dict[str, Any]
    explanation: dict[str, Any]
    route: str
    route_reason: str
    retrieved_evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    groundedness: dict[str, Any]
    citation_repair: dict[str, Any]
    response: dict[str, Any]
    errors: list[str]
