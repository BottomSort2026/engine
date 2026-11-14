from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import model


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=model.N_FEATURES, max_length=model.N_FEATURES
    )


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()  # fails fast at startup if model.joblib is missing
    yield


app = FastAPI(title="ML model API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        label, probabilities = model.predict(req.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    return PredictResponse(prediction=label, probabilities=probabilities)


@app.post("/embed", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        label, probabilities = model.predict(req.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="embed failed") from exc
    return PredictResponse(prediction=label, probabilities=probabilities)
