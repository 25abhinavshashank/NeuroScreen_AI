from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.common import RouteType, SafetyStatus
from backend.app.schemas.groundedness import AuditResult, GroundednessResult
from backend.app.schemas.prediction import Decision, Distribution, ModelInfo, Prediction, Uncertainty
from backend.app.schemas.rag import Citation, EvidenceItem


class AssessmentRequest(BaseModel):
    """Q-CHAT-10-style child screening questionnaire input.

    A1-A10 are binary questionnaire item scores (0/1). Categorical fields
    accept the raw values used by the training dataset (e.g. "m"/"f",
    "yes"/"no"); unknown values are handled by the model's preprocessing
    pipeline rather than rejected here.
    """

    A1_Score: int | None = Field(None, ge=0, le=1)
    A2_Score: int | None = Field(None, ge=0, le=1)
    A3_Score: int | None = Field(None, ge=0, le=1)
    A4_Score: int | None = Field(None, ge=0, le=1)
    A5_Score: int | None = Field(None, ge=0, le=1)
    A6_Score: int | None = Field(None, ge=0, le=1)
    A7_Score: int | None = Field(None, ge=0, le=1)
    A8_Score: int | None = Field(None, ge=0, le=1)
    A9_Score: int | None = Field(None, ge=0, le=1)
    A10_Score: int | None = Field(None, ge=0, le=1)
    age: float | None = Field(None, ge=0, le=120)
    gender: str | None = None
    ethnicity: str | None = None
    jundice: str | None = None
    austim: str | None = None
    contry_of_res: str | None = None
    used_app_before: str | None = None
    relation: str | None = None

    @field_validator(
        "gender",
        "ethnicity",
        "jundice",
        "austim",
        "contry_of_res",
        "used_app_before",
        "relation",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AssessmentResponse(BaseModel):
    model_config = {"extra": "allow"}

    summary: str
    prediction: Prediction
    model: ModelInfo
    uncertainty: Uncertainty
    distribution: Distribution
    decision: Decision
    route: RouteType
    route_reason: str
    evidence: list[EvidenceItem]
    citations: list[Citation]
    safety_status: SafetyStatus
    safety_notes: list[str]
    llm_explanation: str
    llm_explanation_original: str = ""
    generator: str
    fallback_used: bool = False
    final_explanation_status: str = ""
    groundedness: GroundednessResult
    groundedness_passed: bool
    audit: AuditResult
    disclaimer: str
