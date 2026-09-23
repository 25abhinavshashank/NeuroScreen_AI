from __future__ import annotations

from backend.app.db.repositories.screening import create_screening_record
from backend.app.tests.api.test_history import SAMPLE_RESPONSE


def test_monitoring_endpoints_empty_state(client):
    for path in (
        "/api/v1/monitoring/performance",
        "/api/v1/monitoring/fairness",
        "/api/v1/monitoring/drift",
        "/api/v1/monitoring/audit",
    ):
        r = client.get(path)
        assert r.status_code == 200


def test_performance_reflects_recorded_screenings(client, db_session):
    create_screening_record(db_session, SAMPLE_RESPONSE, {"gender": "m"}, model_version="v2.0.0")
    create_screening_record(db_session, SAMPLE_RESPONSE, {"gender": "f"}, model_version="v2.0.0")

    r = client.get("/api/v1/monitoring/performance")
    body = r.json()
    assert body["total_screenings"] == 2
    assert body["label_distribution"] == {"YES": 2}


def test_fairness_breaks_down_by_gender(client, db_session):
    create_screening_record(db_session, SAMPLE_RESPONSE, {"gender": "m"}, model_version="v2.0.0")

    r = client.get("/api/v1/monitoring/fairness")
    body = r.json()
    assert body["breakdown"]["gender"]["m"]["n"] == 1
    assert "disclaimer" in body


def test_audit_trail_lists_entries(client, db_session):
    create_screening_record(db_session, SAMPLE_RESPONSE, {"gender": "m"}, model_version="v2.0.0")

    r = client.get("/api/v1/monitoring/audit")
    body = r.json()
    assert body["total"] == 1
    assert body["entries"][0]["prediction_label"] == "YES"
