# Considerations: FLEX models in engine

This change replaces the template `/predict` with the two FLEX droplet models (PSI-69), exported by the `train`
repo, and adds `GET /model-info`. It was written without running it, so read this before relying on it.

## Status

- **Not run yet.** The code compiles, and `app/inference.py` is identical to the train copy, but neither
  `pytest` nor `docker build` has run. Verify it first:

  ```bash
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pytest
  # or: docker build -t engine .
  ```

- **If the service does not start, `/embed` is down too**, and the MCP's corpus search with it: the models load
  in the same startup as the embedding model. To unblock the team quickly, revert this commit (`git revert`)
  while it is fixed.

## What changed

| Item | Change |
|---|---|
| `models/flex-outcome-0.2/`, `models/flex-extinction-0.2/` | The two exported model folders (about 200 KB), committed and baked into the image by `COPY . .`. No volumes |
| `app/inference.py` | **Verbatim copy** of `train/src/flame/training/inference.py`. Do not edit it here: change it in train, then copy it |
| `app/flex.py` | Loads both models at startup; `predict` runs both and adds `fuel_burned_fraction`; `model_info` builds `/model-info` |
| `app/main.py` | `POST /predict` and `GET /model-info` with Pydantic schemas (published in `/openapi.json`); `/health` also reports the model versions |
| `app/model.py`, `predict.joblib` | Removed (template demo) |
| `requirements.txt` | scikit-learn **1.9.1 → 1.7.2**, plus numpy 2.3.5, pandas 2.3.3, scipy 1.18.1, joblib 1.6.0 |
| `.gitignore` | No longer ignores `model.joblib` (the model folders are committed) |
| `tests/test_predict.py` | New: the model cards' reference queries, invalid requests (422), `/model-info`, startup failures |
| `tests/test_api.py` | Template predict tests removed; `/health` expects the `models` field |

## Library versions

- The pickled models only load correctly with the versions they were exported with
  (`model_info.json["versions"]` of each folder). That is why scikit-learn goes **down** to 1.7.2.
- **The service refuses to start** if the installed scikit-learn or numpy differ from `model_info.json`. The
  error names the library and the expected version.
- If `pip install` fails (for example it cannot find `scipy==1.18.1` or `joblib==1.6.0`, or they conflict with
  torch / sentence-transformers), **do not change the pins on your own**. They must stay equal to the train
  environment. Report the error so both sides change together, or the models get re-exported.

## Tests

- `tests/test_predict.py` expects the exact reference queries of the train model cards
  (`reports/flex-outcome-0.2_model_card.md`, `reports/flex-extinction-0.2_model_card.md`): probabilities,
  ratios, `support_n`, confidence and the nearest tests.
- **If a number differs, report it; do not adjust the test.** A difference means engine does not answer what
  the models were validated with: a stale copy of `inference.py`, other library versions or other model files.
- The API tests start the whole app, so the first run also downloads the embedding model (as before).

## Startup failures

| Message starts with | Cause | Fix |
|---|---|---|
| `Missing model folder` | `models/<folder>` is missing | Copy it from `train/artifacts/models/`, or set `MODELS_DIR` |
| `... was exported with scikit-learn X, but Y is installed` (or numpy) | Version mismatch | Install the versions in `requirements.txt` |
| `... holds <version> (<task>); expected ...` | A folder holds another model | Copy the right folder, or update `MODEL_FOLDERS` in `app/flex.py` |
| A Pydantic validation error at startup | The models' output no longer matches the response schema | Report it: the train contract changed |

## Contract

- Request and response: `README.md` (sections `POST /predict` and `GET /model-info`), and `/openapi.json`, the
  source of truth. **The MCP should regenerate its TypeScript types from `/openapi.json`.**
- `POST /predict` body: `fuel`, `pressure_atm`, `o2`, `suppressant`, `suppressant_fraction`, `droplet_d0_mm`.
  A value outside its tested range is **not** an error: it shows up in `out_of_range_fields` and the confidence
  becomes `low`.
- 422 has two shapes. Schema errors come from FastAPI (`detail` is a list); the models' own rules (unknown fuel,
  fraction without a suppressant, `o2 + suppressant_fraction > 1`) return `detail` as one string.
- The extinction block is always computed. Consumers show it only when `outcome.outcome` is `extinction`.

## Updating the models later

Export them in train, copy both folders into `models/` (update `MODEL_FOLDERS` in `app/flex.py` if the version
changed), update the expected values in `tests/test_predict.py` from the new model cards, copy `inference.py` if it
changed, and rebuild the image.
