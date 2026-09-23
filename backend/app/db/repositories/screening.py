from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import ScreeningRecord


def create_screening_record(
    db: Session,
    response: dict[str, Any],
    patient_data: dict[str, Any],
    model_version: str = "unknown",
) -> ScreeningRecord:
    prediction = response.get("prediction", {})
    uncertainty = response.get("uncertainty", {})
    distribution = response.get("distribution", {})
    decision = response.get("decision", {})
    groundedness = response.get("groundedness", {})
    audit = response.get("audit", {})

    record = ScreeningRecord(
        model_version=model_version,
        prediction_label=prediction.get("label", "UNKNOWN"),
        probability_yes=float(prediction.get("probability_yes", 0.0)),
        uncertainty_category=uncertainty.get("category", "UNKNOWN"),
        distribution_status=distribution.get("status", "UNKNOWN"),
        decision_status=decision.get("status", "UNKNOWN"),
        route=response.get("route", "UNKNOWN"),
        safety_status=response.get("safety_status", "UNKNOWN"),
        groundedness_score=float(groundedness.get("groundedness_score", 0.0)),
        groundedness_passed=bool(response.get("groundedness_passed", False)),
        fallback_used=bool(response.get("fallback_used", False)),
        audit_status=audit.get("final_status", "UNKNOWN"),
        llm_explanation=response.get("llm_explanation", ""),
        patient_data=patient_data,
        full_response=response,
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_screening_record(db: Session, record_id: str) -> ScreeningRecord | None:
    return db.get(ScreeningRecord, record_id)


def list_screening_records(
    db: Session,
    limit: int = 20,
    offset: int = 0,
    route: str | None = None,
) -> tuple[list[ScreeningRecord], int]:
    query = select(ScreeningRecord)
    count_query = select(func.count()).select_from(ScreeningRecord)

    if route:
        query = query.where(ScreeningRecord.route == route)
        count_query = count_query.where(ScreeningRecord.route == route)

    total = db.execute(count_query).scalar_one()

    records = (
        db.execute(
            query.order_by(ScreeningRecord.created_at.desc()).limit(limit).offset(offset)
        )
        .scalars()
        .all()
    )

    return list(records), total
