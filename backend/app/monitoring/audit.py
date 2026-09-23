from __future__ import annotations

from backend.app.db.models import ScreeningRecord


def build_audit_entry(record: ScreeningRecord) -> dict:
    """Flatten one screening record into a single audit-log line.

    Covers everything the synopsis's audit trail asked for: model
    version, route, prediction, groundedness, and final decision,
    each timestamped — all of it is already captured per-record by
    `ScreeningRecord`, so this is a formatting layer, not new tracking.
    """

    return {
        "id": record.id,
        "timestamp": record.created_at.isoformat(),
        "model_version": record.model_version,
        "route": record.route,
        "safety_status": record.safety_status,
        "prediction_label": record.prediction_label,
        "probability_yes": record.probability_yes,
        "distribution_status": record.distribution_status,
        "decision_status": record.decision_status,
        "groundedness_score": record.groundedness_score,
        "groundedness_passed": record.groundedness_passed,
        "fallback_used": record.fallback_used,
        "audit_status": record.audit_status,
    }


def build_audit_trail(records: list[ScreeningRecord]) -> list[dict]:
    return [build_audit_entry(record) for record in records]
