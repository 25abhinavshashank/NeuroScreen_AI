from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PredictedLabel(str, Enum):
    YES = "YES"
    NO = "NO"


class UncertaintyCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DistributionStatus(str, Enum):
    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    OOD_NOT_AVAILABLE = "OOD_NOT_AVAILABLE"


class DecisionStatus(str, Enum):
    SCREENING_RESULT = "SCREENING_RESULT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    OOD_REVIEW = "OOD_REVIEW"


class ModelInfo(BaseModel):
    name: str
    version: str
    status: str
    age_band: str


class Prediction(BaseModel):
    label: PredictedLabel
    probability_yes: float = Field(ge=0.0, le=1.0)
    probability_no: float = Field(ge=0.0, le=1.0)


class Uncertainty(BaseModel):
    confidence: float
    uncertainty: float
    category: UncertaintyCategory


class Distribution(BaseModel):
    status: DistributionStatus
    distance: float | None = None
    threshold: float | None = None
    is_ood: bool
    age_out_of_range: bool = False


class Decision(BaseModel):
    status: DecisionStatus
    reason: str
    uncertainty_category: UncertaintyCategory
    confidence: float


class FeatureContribution(BaseModel):
    model_config = {"extra": "allow"}

    feature: str
    shap_value: float
    direction: str


class Explanation(BaseModel):
    model_config = {"extra": "allow"}

    available: bool = True
    output_scale: str | None = None
    note: str | None = None
    top_contributions: list[FeatureContribution] = Field(default_factory=list)


class PredictionMetadata(BaseModel):
    feature_count: int
    feature_names: list[str]
    excluded_features: list[str]
    transformed_feature_count: int


class PredictionResult(BaseModel):
    """Mirrors the dict returned by PredictionService.predict()."""

    model_config = {"extra": "allow"}

    model: ModelInfo
    prediction: Prediction
    uncertainty: Uncertainty
    distribution: Distribution
    decision: Decision
    explanation: Explanation
    metadata: PredictionMetadata
    disclaimer: str
