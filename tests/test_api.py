import pytest
from fastapi.testclient import TestClient

VALID = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
         "ca": 0, "thal": 6}


@pytest.fixture
def client(monkeypatch, model_path):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from src.api import app
    with TestClient(app) as c:  # context manager triggers lifespan (model load)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert "X-Request-ID" in r.headers


@pytest.mark.parametrize("field,value", [("age", -5), ("sex", 3), ("thal", 5), ("cp", 0)])
def test_predict_rejects_invalid_values(client, field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "chol"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=VALID)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_predictions_total" in text
    assert "api_request_latency_seconds" in text
