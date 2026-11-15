"""GET /health. Needs corpus/manifest.json; the first run downloads the embedding model."""
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
            "models": {"outcome": "flex-outcome-0.2", "extinction": "flex-extinction-0.2"},
        }
