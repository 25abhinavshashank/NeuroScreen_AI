from __future__ import annotations

from collections import defaultdict

from backend.app.db.models import ScreeningRecord

DISCLAIMER = (
    "Descriptive subgroup breakdown of live screening traffic only. "
    "This is NOT a validated fairness audit: group sizes here are "
    "typically far too small for statistical significance, and the "
    "underlying training data has a known label-reconstruction issue "
    "(the target is tightly determined by the questionnaire score "
    "itself), so subgroup rate differences may reflect who is using "
    "the tool rather than model bias. Treat as a monitoring signal to "
    "investigate, not a conclusion."
)

SUBGROUP_FIELDS = ("gender", "ethnicity", "jundice", "austim")


def compute_fairness_summary(records: list[ScreeningRecord]) -> dict:
    breakdown: dict[str, dict[str, dict]] = {field: {} for field in SUBGROUP_FIELDS}

    groups: dict[str, dict[str, list[ScreeningRecord]]] = {
        field: defaultdict(list) for field in SUBGROUP_FIELDS
    }

    for record in records:
        patient_data = record.patient_data or {}
        for field in SUBGROUP_FIELDS:
            value = patient_data.get(field)
            key = str(value).strip() if value not in (None, "") else "unknown"
            groups[field][key].append(record)

    for field, subgroups in groups.items():
        for value, group_records in subgroups.items():
            n = len(group_records)
            yes_count = sum(1 for r in group_records if r.prediction_label == "YES")
            breakdown[field][value] = {
                "n": n,
                "positive_rate": round(yes_count / n, 4) if n else None,
                "avg_groundedness_score": round(
                    sum(r.groundedness_score for r in group_records) / n, 4
                )
                if n
                else None,
            }

    return {
        "disclaimer": DISCLAIMER,
        "total_screenings": len(records),
        "breakdown": breakdown,
    }
