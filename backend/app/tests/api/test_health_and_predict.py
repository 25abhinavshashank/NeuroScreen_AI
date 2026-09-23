from __future__ import annotations

from backend.app.tests.conftest import SAMPLE_ASSESSMENT_REQUEST


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "NEUROSCREEN AI"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "child" in body["models"]
    assert "toddler" in body["models"]
    assert body["models"]["child"]["version"]


def test_models_registry(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["active_models"]["child"]["name"]
    assert body["active_models"]["toddler"]["age_range_years"] == [1.0, 3.0]


def test_predict_ml_only(client):
    r = client.post("/api/v1/predict", json=SAMPLE_ASSESSMENT_REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["label"] in ("YES", "NO")
    assert 0.0 <= body["prediction"]["probability_yes"] <= 1.0


def test_predict_rejects_out_of_range_score(client):
    bad_payload = {**SAMPLE_ASSESSMENT_REQUEST, "A1_Score": 5}
    r = client.post("/api/v1/predict", json=bad_payload)
    assert r.status_code == 422


def test_unhandled_error_returns_structured_500(client, monkeypatch):
    from backend.app.api.routes import predict as predict_route

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(predict_route.prediction_service, "predict", _boom)

    r = client.post("/api/v1/predict", json=SAMPLE_ASSESSMENT_REQUEST)
    assert r.status_code == 500
    assert "Prediction failed" in r.json()["detail"]
