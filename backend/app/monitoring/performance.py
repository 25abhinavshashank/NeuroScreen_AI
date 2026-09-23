from __future__ import annotations

from collections import Counter

from backend.app.db.models import ScreeningRecord


def compute_performance_summary(records: list[ScreeningRecord]) -> dict:
    """Operational performance over recorded live screenings.

    This is NOT a held-out accuracy/precision/recall/F1/ROC-AUC report
    against ground truth — that offline evaluation belongs to
    `scripts/evaluate.py` and the metrics already recorded under
    `models/metadata/*.csv`, since live requests have no ground-truth
    label to score against. This reports what the running system is
    actually doing: label mix, workflow health, and response quality
    signals, which is what "monitoring" (as opposed to "evaluation")
    means for a system with no online ground truth.
    """

    total = len(records)

    if total == 0:
        return {
            "total_screenings": 0,
            "label_distribution": {},
            "route_distribution": {},
            "decision_distribution": {},
            "avg_probability_yes": None,
            "avg_groundedness_score": None,
            "groundedness_pass_rate": None,
            "fallback_rate": None,
            "audit_pass_rate": None,
        }

    label_counts = Counter(r.prediction_label for r in records)
    route_counts = Counter(r.route for r in records)
    decision_counts = Counter(r.decision_status for r in records)

    return {
        "total_screenings": total,
        "label_distribution": dict(label_counts),
        "route_distribution": dict(route_counts),
        "decision_distribution": dict(decision_counts),
        "avg_probability_yes": round(
            sum(r.probability_yes for r in records) / total, 4
        ),
        "avg_groundedness_score": round(
            sum(r.groundedness_score for r in records) / total, 4
        ),
        "groundedness_pass_rate": round(
            sum(1 for r in records if r.groundedness_passed) / total, 4
        ),
        "fallback_rate": round(
            sum(1 for r in records if r.fallback_used) / total, 4
        ),
        "audit_pass_rate": round(
            sum(1 for r in records if r.audit_status == "PASSED") / total, 4
        ),
    }
