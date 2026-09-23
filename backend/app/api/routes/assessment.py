from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.agents.graph import neuroscreen_graph
from backend.app.core.logging import get_logger
from backend.app.db.database import get_db
from backend.app.db.repositories.screening import create_screening_record
from backend.app.schemas.assessment import AssessmentRequest, AssessmentResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/assessment", tags=["Assessment"])

@router.post("", response_model=AssessmentResponse)
def create_assessment(request: AssessmentRequest, db: Annotated[Session, Depends(get_db)]):
    try:
        patient_data = request.model_dump(exclude_none=False)
        state = {
            "patient_data": patient_data,
            "query": "Explain general developmental screening and what a positive screening result means for follow-up assessment.",
            "errors": [],
        }
        final_state = neuroscreen_graph.invoke(state)
        response = final_state.get("response")
        if not response:
            raise RuntimeError("Workflow returned no response.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Assessment workflow failed")
        raise HTTPException(status_code=500, detail=f"Assessment workflow failed: {exc}") from exc

    try:
        model_version = response.get("model", {}).get("version", "unknown")
        create_screening_record(db, response, patient_data, model_version=model_version)
    except Exception:
        # History persistence is a side effect, not the point of this
        # request: a DB/write failure shouldn't turn a valid screening
        # result into a 500 for the caller.
        logger.exception("Failed to persist screening record to history")

    return response
