"""FastAPI model-serving application with request logging and Prometheus metrics."""
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import FEATURES, MODEL_PATH

# ----------------------------------------------------------------- logging
LOG_FILE = os.getenv("LOG_FILE", "logs/api.log")
handlers = [logging.StreamHandler(sys.stdout)]
try:
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(LOG_FILE))
except OSError:  # read-only filesystem (e.g. hardened container) -> stdout only
    pass
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=handlers)
logger = logging.getLogger("heart-api")

# ----------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("api_predictions_total", "Predictions served", ["label"])
ERRORS = Counter("api_errors_total", "Unhandled errors", ["endpoint"])
MODEL_LOADED = Gauge("api_model_loaded", "1 if the model is loaded")
KNOWN_ENDPOINTS = {"/", "/health", "/predict", "/metrics", "/model-info", "/docs", "/openapi.json"}


# ----------------------------------------------------------------- schemas
class PatientFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy")
    thal: int = Field(..., description="3 = normal, 6 = fixed defect, 7 = reversible defect")

    @field_validator("thal")
    @classmethod
    def thal_allowed(cls, v: int) -> int:
        if v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    request_id: str


# ----------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    logger.info("Loading model from %s", path)
    app.state.model = joblib.load(path)  # fail fast if the artifact is missing
    MODEL_LOADED.set(1)
    yield
    MODEL_LOADED.set(0)


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    path = request.url.path if request.url.path in KNOWN_ENDPOINTS else "other"
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        ERRORS.labels(endpoint=path).inc()
        logger.exception("unhandled error request_id=%s", request_id)
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        logger.info(json.dumps({"event": "request", "request_id": request_id,
                                "method": request.method, "path": path,
                                "status": status, "latency_ms": round(elapsed * 1000, 2)}))


@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.get("/model-info")
def model_info():
    return {"features": FEATURES, "model_path": os.getenv("MODEL_PATH", str(MODEL_PATH))}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    frame = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
    p_disease = float(model.predict_proba(frame)[0][1])
    pred = int(p_disease >= 0.5)
    confidence = p_disease if pred == 1 else 1 - p_disease
    label = "heart disease likely" if pred else "no heart disease"

    PREDICTIONS.labels(label=str(pred)).inc()
    logger.info(json.dumps({"event": "prediction", "request_id": request.state.request_id,
                            "input": patient.model_dump(), "prediction": pred,
                            "probability_disease": round(p_disease, 4)}))
    return PredictionResponse(prediction=pred, label=label, confidence=round(confidence, 4),
                              probability_disease=round(p_disease, 4),
                              request_id=request.state.request_id)
