from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.models import ScreeningRecord
from backend.app.monitoring.audit import build_audit_trail
from backend.app.monitoring.drift import compute_drift_summary
from backend.app.monitoring.fairness import compute_fairness_summary
from backend.app.monitoring.performance import compute_performance_summary
from backend.app.schemas.monitoring import AuditTrail, DriftSummary, FairnessSummary, PerformanceSummary

router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])


def _recent_records(db: Session, limit: int) -> list[ScreeningRecord]:
    return list(
        db.execute(
            select(ScreeningRecord).order_by(ScreeningRecord.created_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )


@router.get("/performance", response_model=PerformanceSummary)
def get_performance(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> PerformanceSummary:
    return compute_performance_summary(_recent_records(db, limit))


@router.get("/fairness", response_model=FairnessSummary)
def get_fairness(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> FairnessSummary:
    return compute_fairness_summary(_recent_records(db, limit))


@router.get("/drift", response_model=DriftSummary)
def get_drift(
    db: Annotated[Session, Depends(get_db)],
    recent_n: Annotated[int, Query(ge=1, le=200)] = 20,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> DriftSummary:
    return compute_drift_summary(_recent_records(db, limit), recent_n=recent_n)


@router.get("/audit", response_model=AuditTrail)
def get_audit_trail(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> AuditTrail:
    records = _recent_records(db, limit)
    return AuditTrail(total=len(records), entries=build_audit_trail(records))
