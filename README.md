# FastAPI ML model template

Minimal FastAPI service around a scikit-learn model. The demo model is iris
classification (4 features), trained by `train.py`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                 # writes model.joblib
uvicorn app.main:app --reload   # docs at http://localhost:8000/docs
pytest
```

## Docker

```bash
docker build -t ml-api .
docker run --rm -p 8000:8000 ml-api
```

The image trains `model.joblib` during build. To ship your own artifact
instead, drop the `model.joblib` line from `.dockerignore`, train locally, and
remove the `RUN python train.py` line from the Dockerfile.

## Predict

```bash
curl -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

```json
{"prediction": "0", "probabilities": {"0": 0.9847, "1": 0.0153, "2": 0.0}}
```

## Using your own model

Replace the body of `app/model.py`: `load()` should load your artifact and
`predict()` should return `(label, {class: probability})`. For PyTorch, save
`state_dict` and rebuild the architecture in `load()`. Keep `N_FEATURES` in
sync with your input. No other file needs to change.
