"""POST /embed. Needs corpus/manifest.json; the first run downloads the embedding model."""
import math

from fastapi.testclient import TestClient

from app.main import app

with TestClient(app) as client:
    def test_embed_returns_normalized_vector_of_manifest_dimension():
        r = client.post("/embed", json={"text": "how does CO2 dilution suppress droplet flames"})
        assert r.status_code == 200
        body = r.json()
        assert body["dimension"] == len(body["embedding"]) == 384
        assert math.isclose(math.sqrt(sum(x * x for x in body["embedding"])), 1.0, rel_tol=1e-3)

    def test_embed_rejects_blank_text():
        assert client.post("/embed", json={"text": "   "}).status_code == 422
        assert client.post("/embed", json={}).status_code == 422
