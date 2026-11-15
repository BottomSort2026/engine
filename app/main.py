from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import embedder, model

MAX_QUERY_CHARS = 2000


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=model.N_FEATURES, max_length=model.N_FEATURES
    )


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class EmbedRequest(BaseModel):
    """A search query, in English (the corpus is in English). Do NOT add the query prefix: /embed adds it."""

    text: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS,
                      examples=["how does CO2 dilution suppress droplet flames"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class EmbedResponse(BaseModel):
    """Query vector for sqlite-vec: L2-normalized float32 values, same space as corpus.db."""

    embedding: list[float]
    dimension: int
    model_id: str
    revision: str | None


class EmbeddingInfo(BaseModel):
    """What /embed serves, from corpus/manifest.json. The MCP reports the same corpus build date."""

    model_id: str
    revision: str | None
    dimension: int
    corpus_built_at: str | None


class HealthResponse(BaseModel):
    status: str
    embedding: EmbeddingInfo


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()  # fails fast at startup if predict.joblib is missing
    embedder.load()  # fails fast at startup if corpus/manifest.json is missing or invalid
    yield


app = FastAPI(title="ML model API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", embedding=EmbeddingInfo(**embedder.info()))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        label, probabilities = model.predict(req.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    return PredictResponse(prediction=label, probabilities=probabilities)


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        vector = embedder.embed(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="embed failed") from exc
    info = embedder.info()
    return EmbedResponse(embedding=vector, dimension=len(vector), model_id=info["model_id"],
                         revision=info["revision"])
