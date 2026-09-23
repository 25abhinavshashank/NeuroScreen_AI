from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from backend.app.schemas.assessment import AssessmentResponse


class ScreeningSummary(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    created_at: datetime
    model_version: str
    prediction_label: str
    probability_yes: float
    uncertainty_category: str
    distribution_status: str
    decision_status: str
    route: str
    safety_status: str
    groundedness_score: float
    groundedness_passed: bool
    fallback_used: bool
    audit_status: str


class ScreeningDetail(ScreeningSummary):
    patient_data: dict
    full_response: AssessmentResponse


class ScreeningHistoryPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ScreeningSummary]
