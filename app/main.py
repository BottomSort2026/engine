from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import embedder, flex

MAX_QUERY_CHARS = 2000


class PredictRequest(BaseModel):
    """The droplet test conditions, the same for both models. GET /model-info gives the values and tested
    ranges. A value outside its tested range is NOT rejected: it is listed in out_of_range_fields and the
    confidence becomes low. A missing or unknown field, an unknown fuel or suppressant, a fraction without a
    suppressant, or o2 + suppressant_fraction > 1 returns 422."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "fuel": "methanol", "pressure_atm": 1.0, "o2": 0.21, "suppressant": "none",
        "suppressant_fraction": 0.0, "droplet_d0_mm": 3.0}]})

    fuel: str = Field(description="Droplet fuel, e.g. methanol or heptane")
    pressure_atm: float = Field(strict=True, allow_inf_nan=False, description="Chamber pressure (atm)")
    o2: float = Field(strict=True, allow_inf_nan=False, description="O2 mole fraction")
    suppressant: str = Field(description="Suppressant gas added to the atmosphere: none, co2 or he")
    suppressant_fraction: float = Field(strict=True, allow_inf_nan=False,
                                        description="Mole fraction of that gas; 0 when suppressant is none")
    droplet_d0_mm: float = Field(strict=True, allow_inf_nan=False, description="Initial droplet diameter (mm)")


class NearestTest(BaseModel):
    """A real ISS test close to the query. testsheet_id is the TestSheet title in corpus.db."""

    cir_id: str | None
    testsheet_id: str | None
    test_end: str | None = Field(description="How the real test ended: extinction, disruption or completion")
    conditions: dict[str, float | str | None] = Field(
        description="That test's fuel, pressure_atm, o2, co2, he and d0_mm")
    distance: float = Field(description="Distance to the query, in standardized units")


class TrainingSource(BaseModel):
    dataset: str
    accession: str
    sha256: str
    rows: int


class Support(BaseModel):
    """How much real data backs the prediction (the model's own k-NN over its training tests)."""

    in_tested_range: bool = Field(
        description="Every input inside its tested range AND the closest real test within the 'near' threshold")
    out_of_range_fields: list[str]
    support_n: int = Field(description="Real tests of the same fuel within the 'near' threshold")
    nearest_tests: list[NearestTest]
    confidence: Literal["low", "medium", "high"]
    model_version: str
    training_source: TrainingSource


class OutcomePrediction(Support):
    outcome: Literal["extinction", "no_extinction"] = Field(
        description="extinction: the flame goes out on its own; no_extinction: the droplet burns out or breaks apart")
    probabilities: dict[str, float]


class ExtinctionPrediction(Support):
    """Meaningful only when outcome.outcome is extinction."""

    extinction_diameter_ratio: float = Field(description="D_ext / D0, in [0, 1]")
    extinction_diameter_mm: float
    fuel_burned_fraction: float = Field(description="1 - (D_ext / D0)^3: share of the droplet volume burned")


class PredictResponse(BaseModel):
    outcome: OutcomePrediction
    extinction: ExtinctionPrediction


class ModelDescription(BaseModel):
    model_version: str
    task: str
    algorithm: str
    inputs: dict[str, dict[str, Any]] = Field(description="This model's own tested ranges")
    outputs: list[str]
    training_source: dict[str, Any]
    metrics: dict[str, float] = Field(description="Cross-validated metrics of the shipped model")
    metrics_unit: str


class ModelInfoResponse(BaseModel):
    inputs: dict[str, dict[str, Any]] = Field(
        description="One control per input, wide enough for both models (by_selector: range per suppressant)")
    outcome: ModelDescription
    extinction: ModelDescription


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
    models: dict[str, str] = Field(description="Model version served by /predict, per model")


@asynccontextmanager
async def lifespan(_: FastAPI):
    flex.load()  # fails fast at startup if a model folder is missing or the library versions differ
    PredictResponse.model_validate(flex.predict(flex.sample_inputs()))  # the models match the response schema
    embedder.load()  # fails fast at startup if corpus/manifest.json is missing or invalid
    yield


app = FastAPI(title="Flame in Freefall engine", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", embedding=EmbeddingInfo(**embedder.info()), models=flex.versions())


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Runs both FLEX models (outcome and extinction diameter), each with its k-NN support."""
    try:
        result = flex.predict(req.model_dump())
    except ValueError as exc:  # invalid inputs (inference.validate_inputs / check_constraints)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    return PredictResponse.model_validate(result)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """The inputs (values, tested ranges, units) and outputs of both models, to build the simulator controls."""
    return ModelInfoResponse.model_validate(flex.model_info())


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        vector = embedder.embed(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="embed failed") from exc
    info = embedder.info()
    return EmbedResponse(embedding=vector, dimension=len(vector), model_id=info["model_id"],
                         revision=info["revision"])
