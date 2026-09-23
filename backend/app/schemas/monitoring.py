from __future__ import annotations

from pydantic import BaseModel


class ActiveModelSummary(BaseModel):
    name: str
    version: str
    status: str
    age_range_years: list[float] | None = None


class HealthStatus(BaseModel):
    status: str
    environment: str
    models: dict[str, ActiveModelSummary]
    rag_index_available: bool


class ModelRegistryEntry(BaseModel):
    name: str
    version: str
    status: str
    artifact: str
    metadata: str
    age_range_years: list[float] | None = None


class ModelRegistryInfo(BaseModel):
    active_models: dict[str, ModelRegistryEntry]
    updated_at: str | None = None


class PerformanceSummary(BaseModel):
    total_screenings: int
    label_distribution: dict[str, int]
    route_distribution: dict[str, int]
    decision_distribution: dict[str, int]
    avg_probability_yes: float | None
    avg_groundedness_score: float | None
    groundedness_pass_rate: float | None
    fallback_rate: float | None
    audit_pass_rate: float | None


class SubgroupStats(BaseModel):
    n: int
    positive_rate: float | None
    avg_groundedness_score: float | None


class FairnessSummary(BaseModel):
    disclaimer: str
    total_screenings: int
    breakdown: dict[str, dict[str, SubgroupStats]]


class DriftSummary(BaseModel):
    total_screenings: int
    recent_window: int
    ood_rate_recent: float | None
    ood_rate_overall: float | None
    avg_probability_recent: float | None
    avg_probability_overall: float | None
    ood_drift_detected: bool
    prediction_drift_detected: bool


class AuditEntry(BaseModel):
    id: str
    timestamp: str
    model_version: str
    route: str
    safety_status: str
    prediction_label: str
    probability_yes: float
    distribution_status: str
    decision_status: str
    groundedness_score: float
    groundedness_passed: bool
    fallback_used: bool
    audit_status: str


class AuditTrail(BaseModel):
    total: int
    entries: list[AuditEntry]
