from __future__ import annotations

from backend.app.db.models import ScreeningRecord

# Absolute-difference thresholds for flagging drift between the recent
# window and the full history. These are coarse, documented defaults
# for a low-volume research prototype, not statistically derived
# control limits — tune them once real traffic volume exists.
OOD_RATE_DRIFT_THRESHOLD = 0.15
PROBABILITY_DRIFT_THRESHOLD = 0.20


def _ood_rate(records: list[ScreeningRecord]) -> float | None:
    if not records:
        return None
    ood_count = sum(1 for r in records if r.distribution_status == "OUT_OF_DISTRIBUTION")
    return round(ood_count / len(records), 4)


def _avg_probability(records: list[ScreeningRecord]) -> float | None:
    if not records:
        return None
    return round(sum(r.probability_yes for r in records) / len(records), 4)


def compute_drift_summary(records: list[ScreeningRecord], recent_n: int = 20) -> dict:
    """Compare recent live traffic against the full recorded history.

    Feature/population drift is approximated by the model's own OOD
    flag rate (`distribution_status`), computed against the training
    distribution's center/MAD at prediction time — this needs no
    separate reference dataset since the model already carries one.
    Prediction drift compares the recent average positive-class
    probability against the full-history average.

    With no labeled ground truth for live traffic, these are directional
    signals ("something about incoming requests changed") rather than
    proof of model degradation.
    """

    if not records:
        return {
            "total_screenings": 0,
            "recent_window": recent_n,
            "ood_rate_recent": None,
            "ood_rate_overall": None,
            "avg_probability_recent": None,
            "avg_probability_overall": None,
            "ood_drift_detected": False,
            "prediction_drift_detected": False,
        }

    ordered = sorted(records, key=lambda r: r.created_at, reverse=True)
    recent = ordered[:recent_n]

    ood_rate_recent = _ood_rate(recent)
    ood_rate_overall = _ood_rate(ordered)
    avg_prob_recent = _avg_probability(recent)
    avg_prob_overall = _avg_probability(ordered)

    ood_drift = (
        ood_rate_recent is not None
        and ood_rate_overall is not None
        and abs(ood_rate_recent - ood_rate_overall) >= OOD_RATE_DRIFT_THRESHOLD
    )

    prediction_drift = (
        avg_prob_recent is not None
        and avg_prob_overall is not None
        and abs(avg_prob_recent - avg_prob_overall) >= PROBABILITY_DRIFT_THRESHOLD
    )

    return {
        "total_screenings": len(records),
        "recent_window": min(recent_n, len(records)),
        "ood_rate_recent": ood_rate_recent,
        "ood_rate_overall": ood_rate_overall,
        "avg_probability_recent": avg_prob_recent,
        "avg_probability_overall": avg_prob_overall,
        "ood_drift_detected": ood_drift,
        "prediction_drift_detected": prediction_drift,
    }
