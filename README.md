# engine: ML service for Flame in Freefall

FastAPI service behind the MCP server's vector search and, later, the dashboard's simulator.

| Endpoint | Status |
|---|---|
| `POST /embed` | Query embeddings for the MCP's vector search over `corpus.db`. The reference contract for the MCP. |
| `GET /health` | Liveness, plus the embedding model and the corpus build date this service was built from. |
| `POST /predict` | **Placeholder.** Still the template's demo classifier (`predict.joblib`, 4 numeric features). It will be replaced once the prediction target is decided; don't build on its request or response. |
| `GET /model-info` | **Does not exist yet.** |

Interactive docs: `http://localhost:8000/docs`.

## corpus/manifest.json

The embedding settings (model, revision, dimension, normalization, max length, query prefix) are not
written in code. They are read from `corpus/manifest.json`, a copy of `train/artifacts/corpus/manifest.json`
produced by the corpus build in the `train` repo. That keeps query vectors in the same space as the vectors
stored in `corpus.db`. **The service refuses to start if the manifest is missing or incomplete.** Set
`MANIFEST_PATH` to read it from somewhere else.

When the corpus is rebuilt, copy its new manifest here and rebuild the image.

`app/embedding.py` is a verbatim copy of the same file in the `train` repo. Change it there, then copy it.

## POST /embed

Send the search query in English (the corpus is in English), without the query prefix: `/embed` adds it.
At most 2000 characters; blank text returns 422.

```bash
curl -s -X POST localhost:8000/embed -H 'content-type: application/json' \
  -d '{"text":"how does CO2 dilution suppress droplet flames"}'
```

```json
{
  "embedding": [-0.0604, 0.0863, -0.0245, "... 384 values"],
  "dimension": 384,
  "model_id": "BAAI/bge-small-en-v1.5",
  "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
}
```

The vector is float32 and L2-normalized, ready to query sqlite-vec.

## GET /health

```bash
curl -s localhost:8000/health
```

```json
{
  "status": "ok",
  "embedding": {
    "model_id": "BAAI/bge-small-en-v1.5",
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "dimension": 384,
    "corpus_built_at": "2026-11-15T00:41:43+00:00"
  }
}
```

The MCP reports the same corpus build date. If `corpus_built_at` differs between the two services, they
were built from different corpora and search results can't be trusted.

## POST /predict (placeholder)

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

```json
{"prediction": "0", "probabilities": {"0": 0.9847, "1": 0.0153, "2": 0.0}}
```

Exactly 4 features; any other count returns 422.

## Run locally

Python 3.12, the same as the image.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest
```

The first start (and the first `pytest` run) downloads the embedding model into the Hugging Face cache
(`~/.cache/huggingface`, or `$HF_HOME`). Later starts load it from there.

## Docker

```bash
docker build -t engine .
docker run --rm -p 8000:8000 engine
```

The build downloads the embedding model into the image (`HF_HOME=/opt/huggingface`), at the revision the
manifest resolves to, and then sets `HF_HUB_OFFLINE=1`. The container starts without network access.
Rebuild the image after changing `corpus/manifest.json`.

## Versions

`requirements.txt` pins `torch`, `sentence-transformers` and `transformers` to the versions the corpus was
built with (change them only together with the `train` environment), and `scikit-learn` to 1.9.1, the
version `predict.joblib` was pickled with.
