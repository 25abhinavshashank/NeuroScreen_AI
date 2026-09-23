
from __future__ import annotations

from backend.app.agents.state import NeuroScreenState
from backend.app.ml.prediction_service import prediction_service


def classifier_node(state: NeuroScreenState) -> dict:
    """
    Run the registered ML model through the existing PredictionService.

    LangGraph is responsible for orchestration.
    PredictionService remains responsible for ML inference,
    uncertainty, OOD detection, decision logic, and SHAP.
    """

    patient_data = state.get("patient_data")

    if not patient_data:
        return {
            "errors": ["No patient assessment data provided."]
        }

    try:
        result = prediction_service.predict(patient_data)

        return {
            "prediction": result.get("prediction", {}),
            "uncertainty": result.get("uncertainty", {}),
            "distribution": result.get("distribution", {}),
            "decision": result.get("decision", {}),
            "explanation": result.get("explanation", {}),
            "model_info": result.get("model", {}),
        }

    except Exception as exc:
        return {
            "errors": [
                f"Classifier node failed: {str(exc)}"
            ]
        }