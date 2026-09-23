from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.repositories.screening import get_screening_record, list_screening_records
from backend.app.schemas.history import ScreeningDetail, ScreeningHistoryPage

router = APIRouter(prefix="/api/v1/history", tags=["History"])


@router.get("", response_model=ScreeningHistoryPage)
def list_history(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    route: str | None = None,
) -> ScreeningHistoryPage:
    records, total = list_screening_records(db, limit=limit, offset=offset, route=route)
    return ScreeningHistoryPage(total=total, limit=limit, offset=offset, items=records)


@router.get("/{record_id}", response_model=ScreeningDetail)
def get_history_item(record_id: str, db: Annotated[Session, Depends(get_db)]) -> ScreeningDetail:
    record = get_screening_record(db, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Screening record not found.")
    return record
