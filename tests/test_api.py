"""GET /health and POST /predict. Needs corpus/manifest.json; the first run downloads the embedding model."""
from fastapi.testclient import TestClient

from app import embedder
from app.main import app

with TestClient(app) as client:
    def test_health_reports_the_manifest_embedding():
        config, manifest = embedder.load_config()
        assert client.get("/health").json() == {
            "status": "ok",
            "embedding": {
                "model_id": config.model_id,
                "revision": config.revision,
                "dimension": config.dimension,
                "corpus_built_at": manifest["built_at"],
            },
        }

    def test_predict():
        r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert r.status_code == 200
        assert r.json()["prediction"] == "0"
        assert abs(sum(r.json()["probabilities"].values()) - 1) < 0.01

    def test_wrong_feature_count():
        assert client.post("/predict", json={"features": [1, 2]}).status_code == 422
