from __future__ import annotations

from pydantic import BaseModel, Field


class GroundednessThresholds(BaseModel):
    groundedness_threshold: float
    claim_support_threshold: float
    max_unsupported_claim_rate: float


class ClaimEvaluation(BaseModel):
    model_config = {"extra": "allow"}

    claim: str
    citation_ids: list[str] = Field(default_factory=list)
    supported: bool
    support_score: float
    matched_evidence: list[str] = Field(default_factory=list)


class GroundednessResult(BaseModel):
    model_config = {"extra": "allow"}

    groundedness_score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    cited_claims: int = 0
    claims: list[ClaimEvaluation] = Field(default_factory=list)
    passed: bool
    thresholds: GroundednessThresholds
    error: str | None = None


class AuditGroundednessSummary(BaseModel):
    score: float
    citation_coverage: float
    citation_precision: float
    unsupported_claim_rate: float
    total_claims: int
    supported_claims: int
    unsupported_claims: int


class AuditChecks(BaseModel):
    non_empty: bool
    diagnostic_claims: bool
    probability_claims: bool
    reliability_claims: bool
    causal_claims: bool
    feature_claims: bool
    individual_interpretation: bool
    citations: bool
    groundedness: bool
    unsupported_claims: bool


class AuditResult(BaseModel):
    llm_status: str
    final_status: str
    fallback_used: bool
    reason: str
    checks: AuditChecks
    matched_pattern: str | None = None
    groundedness: AuditGroundednessSummary
