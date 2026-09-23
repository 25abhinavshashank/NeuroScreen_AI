from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.core.logging import get_logger
from backend.app.ml.prediction_service import prediction_service
from backend.app.schemas.assessment import AssessmentRequest
from backend.app.schemas.prediction import PredictionResult

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResult)
def predict(request: AssessmentRequest) -> PredictionResult:
    """ML-only prediction: model, uncertainty, OOD, decision, SHAP.

    Skips RAG retrieval and LLM explanation so it works without an
    Ollama server and returns fast — useful for the frontend's initial
    result view before evidence/explanation load, and for testing the
    ML pipeline in isolation.
    """

    try:
        patient_data = request.model_dump(exclude_none=False)
        result = prediction_service.predict(patient_data)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ML-only prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
