from fastapi.testclient import TestClient

from app.main import app

# lifespan loads model.joblib, so run `python train.py` first
with TestClient(app) as client:
    def test_health():
        assert client.get("/health").json() == {"status": "ok"}

    def test_predict():
        r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert r.status_code == 200
        assert r.json()["prediction"] == "0"
        assert abs(sum(r.json()["probabilities"].values()) - 1) < 0.01

    def test_wrong_feature_count():
        assert client.post("/predict", json={"features": [1, 2]}).status_code == 422
