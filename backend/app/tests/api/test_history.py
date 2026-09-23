from __future__ import annotations

from backend.app.db.repositories.screening import create_screening_record

SAMPLE_RESPONSE = {
    "summary": "Screening result: YES.",
    "prediction": {"label": "YES", "probability_yes": 0.9, "probability_no": 0.1},
    "model": {"name": "neuro_screen_logistic_regression", "version": "v2.0.0", "status": "DEMO_ACTIVE", "age_band": "child"},
    "uncertainty": {"confidence": 0.9, "uncertainty": 0.1, "category": "LOW"},
    "distribution": {"status": "IN_DISTRIBUTION", "distance": 0.1, "threshold": 0.25, "is_ood": False},
    "decision": {"status": "SCREENING_RESULT", "reason": "ok", "uncertainty_category": "LOW", "confidence": 0.9},
    "route": "STANDARD_REVIEW",
    "route_reason": "ok",
    "evidence": [],
    "citations": [],
    "safety_status": "SCREENING_ONLY",
    "safety_notes": [],
    "llm_explanation": "Some explanation. [E1]",
    "generator": "test",
    "fallback_used": False,
    "groundedness": {
        "groundedness_score": 1.0,
        "citation_coverage": 1.0,
        "citation_precision": 1.0,
        "unsupported_claim_rate": 0.0,
        "total_claims": 1,
        "supported_claims": 1,
        "unsupported_claims": 0,
        "cited_claims": 1,
        "claims": [],
        "passed": True,
        "thresholds": {
            "groundedness_threshold": 0.7,
            "claim_support_threshold": 0.35,
            "max_unsupported_claim_rate": 0.3,
        },
    },
    "groundedness_passed": True,
    "audit": {
        "llm_status": "PASSED",
        "final_status": "PASSED",
        "fallback_used": False,
        "reason": "ok",
        "checks": {k: True for k in (
            "non_empty", "diagnostic_claims", "probability_claims", "reliability_claims",
            "causal_claims", "feature_claims", "individual_interpretation", "citations",
            "groundedness", "unsupported_claims",
        )},
        "groundedness": {
            "score": 1.0, "citation_coverage": 1.0, "citation_precision": 1.0,
            "unsupported_claim_rate": 0.0, "total_claims": 1, "supported_claims": 1,
            "unsupported_claims": 0,
        },
    },
    "disclaimer": "not a diagnosis",
}


def test_history_empty_by_default(client):
    r = client.get("/api/v1/history")
    assert r.status_code == 200
    assert r.json() == {"total": 0, "limit": 20, "offset": 0, "items": []}


def test_history_lists_created_record(client, db_session):
    create_screening_record(db_session, SAMPLE_RESPONSE, {"age": 8}, model_version="v2.0.0")

    r = client.get("/api/v1/history")
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["prediction_label"] == "YES"
    assert body["items"][0]["model_version"] == "v2.0.0"


def test_history_detail_includes_full_response(client, db_session):
    record = create_screening_record(db_session, SAMPLE_RESPONSE, {"age": 8}, model_version="v2.0.0")

    r = client.get(f"/api/v1/history/{record.id}")
    assert r.status_code == 200
    assert r.json()["full_response"]["llm_explanation"] == "Some explanation. [E1]"


def test_history_detail_404_for_unknown_id(client):
    r = client.get("/api/v1/history/does-not-exist")
    assert r.status_code == 404
