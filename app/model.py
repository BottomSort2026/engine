"""Model loading and inference. Swap joblib for torch.load here to use a DL model."""

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "predict.joblib"

N_FEATURES = 4

_model = None

def load(path: Path = MODEL_PATH) -> None:
    global _model
    _model = joblib.load(path)


def predict(features: list[float]) -> tuple[str, dict[str, float]]:
    if _model is None:
        raise RuntimeError("model not loaded")
    X = [features]
    label = _model.predict(X)[0]
    probs = _model.predict_proba(X)[0]
    return str(label), {
        str(c): round(float(p), 4) for c, p in zip(_model.classes_, probs)
    }
