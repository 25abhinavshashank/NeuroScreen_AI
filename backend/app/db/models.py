from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.database import Base


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ScreeningRecord(Base):
    """One completed run of the screening workflow.

    Stores both queryable summary columns (for the history list and
    future monitoring aggregation) and the full response payload (for
    the history detail view and audit trail), so nothing the workflow
    produced needs to be recomputed to look it up later.
    """

    __tablename__ = "screening_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    model_version: Mapped[str] = mapped_column(String(32))

    prediction_label: Mapped[str] = mapped_column(String(8))
    probability_yes: Mapped[float] = mapped_column(Float)
    uncertainty_category: Mapped[str] = mapped_column(String(16))
    distribution_status: Mapped[str] = mapped_column(String(32))
    decision_status: Mapped[str] = mapped_column(String(32))

    route: Mapped[str] = mapped_column(String(32), index=True)
    safety_status: Mapped[str] = mapped_column(String(32))

    groundedness_score: Mapped[float] = mapped_column(Float)
    groundedness_passed: Mapped[bool] = mapped_column(Boolean)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    audit_status: Mapped[str] = mapped_column(String(32))

    llm_explanation: Mapped[str] = mapped_column(Text)

    patient_data: Mapped[dict] = mapped_column(JSON)
    full_response: Mapped[dict] = mapped_column(JSON)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ScreeningRecord {self.id} route={self.route} label={self.prediction_label}>"
