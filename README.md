# engine: ML service for Flame in Freefall

FastAPI service behind the MCP server's vector search and the Flame Lab predictions.

| Endpoint | Status |
|---|---|
| `POST /embed` | Query embeddings for the MCP's vector search over `corpus.db`. The reference contract for the MCP. |
| `POST /predict` | Runs both FLEX droplet models (PSI-69) on one set of test conditions, each with its k-NN support. |
| `GET /model-info` | Inputs (values, tested ranges, units) and outputs of both models, to build the simulator controls. |
| `GET /health` | Liveness, plus the embedding model, the corpus build date and the model versions this service serves. |

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
  },
  "models": {"outcome": "flex-outcome-0.2", "extinction": "flex-extinction-0.2"}
}
```

The MCP reports the same corpus build date. If `corpus_built_at` differs between the two services, they
were built from different corpora and search results can't be trusted.

## models/: the FLEX models

Two folders exported by the train repo (`python main.py train export --config flex-outcome` and
`--config flex-extinction`), copied from `train/artifacts/models/`. They are committed and baked into the
image (about 200 KB, no volumes):

| Folder | Question | Shipped model |
|---|---|---|
| `models/flex-outcome-0.2/` | Does the droplet flame go out on its own? (`extinction` / `no_extinction`) | Logistic regression |
| `models/flex-extinction-0.2/` | If it goes out, what is D_ext/D0? | Gradient boosting |

Each folder holds `model.joblib`, `support.joblib` (the k-NN over the real ISS tests), `support_reference.csv`
and `model_info.json`. The metrics, selection and reference queries are in the model cards of the train repo
(`reports/flex-*-0.2_model_card.md`).

`app/inference.py` is a **verbatim copy** of `train/src/flame/training/inference.py`, like `app/embedding.py`.
Change it there, then copy it. `tests/test_predict.py` checks that engine returns the model cards' reference
queries, so a stale copy, a different library version or a different model folder fails the tests.

**The service refuses to start** if a model folder is missing (set `MODELS_DIR` to read them from somewhere
else), or if the installed scikit-learn or numpy differ from `model_info.json["versions"]`.

To ship new models: export them in train, copy both folders here (rename `MODEL_FOLDERS` in `app/flex.py` if
the version changed), update the expected values in `tests/test_predict.py` from the new model cards, rebuild.

## POST /predict

One request runs both models on the same conditions:

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"fuel":"methanol","pressure_atm":1.0,"o2":0.21,"suppressant":"none","suppressant_fraction":0.0,"droplet_d0_mm":3.0}'
```

Each model answers in one block with three parts: **`prediction`** (what the model predicts), **`support`**
(how much real data backs it: the model's own k-NN) and **`model`** (which model answered, to cite it).

```json
{
  "outcome": {
    "prediction": {
      "label": "extinction",
      "probabilities": {"extinction": 0.587, "no_extinction": 0.413}
    },
    "support": {
      "in_tested_range": true,
      "out_of_range_fields": [],
      "support_n": 10,
      "threshold": 0.830,
      "confidence": "medium",
      "nearest_tests": [
        {"cir_id": "205R001", "testsheet_id": "FLEX-046", "test_end": "disruption",
         "conditions": {"fuel": "methanol", "pressure_atm": 1.0125, "o2": 0.21, "co2": 0.0, "he": 0.0, "d0_mm": 2.78},
         "distance": 0.361},
        "… 4 more, closest first"
      ]
    },
    "model": {
      "model_version": "flex-outcome-0.2",
      "training_source": {"dataset": "flex_tests", "accession": "PSI-69", "sha256": "abb213…", "rows": 251}
    }
  },
  "extinction": {
    "prediction": {
      "extinction_diameter_ratio": 0.445,
      "extinction_diameter_mm": 1.336,
      "fuel_burned_fraction": 0.912
    },
    "support": {"in_tested_range": true, "out_of_range_fields": [], "support_n": 11, "threshold": 1.016,
                "confidence": "high", "nearest_tests": ["…"]},
    "model": {"model_version": "flex-extinction-0.2", "training_source": {"…": "rows 156"}}
  }
}
```

(Shortened, with rounded numbers; the API returns full floats. The full schema is in `/openapi.json`.)

- **Show the extinction block only when `outcome.prediction.label` is `extinction`.** Otherwise the droplet
  burned out or broke apart, and a diameter at extinction has no meaning. Both blocks are always computed.
- `fuel_burned_fraction = 1 − (D_ext/D0)³`, the share of the droplet volume burned before the flame went out.
- `support`, per block (each model has its own k-NN, over the tests of the queried fuel only):

  | Field | Meaning |
  |---|---|
  | `nearest_tests` | The 5 closest real ISS tests, closest first. `testsheet_id` is the TestSheet title in `corpus.db`; `test_end` is how that real test ended; `conditions` are its real conditions |
  | `distance` | Euclidean distance over pressure, O₂, CO₂, He and D₀, each divided by its standard deviation in that fuel's training tests. 0 = identical conditions |
  | `threshold` | The "near" distance for this model and fuel: the 95th percentile of the distance from each training test to its nearest other test. Compare each `distance` with it |
  | `support_n` | Real tests within `threshold` |
  | `in_tested_range` | Every input inside its own tested range **and** the closest test within `threshold` |
  | `confidence` | `low`: out of range or `support_n` < 3; `high`: in range, `support_n` ≥ 5 and (classifier) probability ≥ 0.7; otherwise `medium` |

  A high probability with `support_n: 0` is an extrapolation, not a supported prediction. Distances of the two
  blocks are not comparable: each k-NN has its own scale and threshold.
- **A value outside its tested range is not an error:** it is listed in `out_of_range_fields` and the
  confidence becomes `low`.
- **422** for a missing or unknown field, a number sent as a string, an unknown fuel or suppressant,
  `suppressant_fraction` ≠ 0 with `suppressant: "none"`, or `o2 + suppressant_fraction > 1`. Schema errors
  come from FastAPI (`detail` is a list); the model's own rules return `detail` as one string.

## GET /model-info

```bash
curl -s localhost:8000/model-info
```

- `inputs`: one control per input, wide enough for both models. `suppressant_fraction.by_selector` gives the
  range for each suppressant (`none`: 0 only).
- `outcome`, `extinction`: each model's own tested ranges, `outputs` (the fields of its `prediction` block),
  training data, cross-validated metrics and k-NN settings (`support.thresholds` per fuel, `nearest_k`, the
  confidence rule).
  The extinction model was trained on fewer tests (O₂ up to 0.30): a query outside its ranges is flagged in its
  block, not rejected.

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
The FLEX model folders in `models/` are copied into the image with the code (`COPY . .`), so no volume is
needed. Rebuild the image after changing `corpus/manifest.json` or `models/`.

## Versions

`requirements.txt` pins `torch`, `sentence-transformers` and `transformers` to the versions the corpus was
built with, and `scikit-learn`, `numpy`, `pandas`, `scipy` and `joblib` to the versions the FLEX models were
exported with (`model_info.json["versions"]`: scikit-learn 1.7.2, numpy 2.3.5). Change them only together
with the `train` environment.
