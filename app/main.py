"""
API REST con FastAPI para servir un modelo de clasificación de diabetes
(Pipeline: imputación de ceros + RandomForestClassifier, dataset Pima).

Endpoints:
    GET  /health      -> estado del servicio y si el modelo está cargado
    GET  /model-info  -> metadatos del modelo (features, clases, métricas)
    GET  /stats       -> contadores de monitoreo básico (solicitudes, errores, latencia)
    POST /predict     -> recibe las 8 variables clínicas y devuelve la clase predicha

Incluye:
    - Validación de entrada con Pydantic (tipos y rangos clínicos razonables)
    - Logging de cada solicitud (request_id, entrada, resultado, latencia) en logs/api.log
    - Manejo de errores con códigos HTTP apropiados y registro de excepciones
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.logging_config import setup_logging

logger = setup_logging()

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "model", "metadata.json")

# Orden y nombres de columnas con los que se entrenó el modelo
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

_model = None
_metadata: Dict = {}
_stats = {
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_requests": 0,
    "errors_4xx": 0,
    "errors_5xx": 0,
    "predictions": 0,
    "predictions_by_class": {"Sin diabetes": 0, "Diabetes": 0},
    "predict_latency_ms_total": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata
    try:
        _model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)
        logger.info("Modelo cargado correctamente desde %s", MODEL_PATH)
    except Exception:
        logger.exception("Error cargando el modelo durante el arranque")
        raise
    yield
    logger.info("Servicio detenido")


app = FastAPI(
    title="Diabetes Classifier API",
    description=(
        "Servicio web que expone un modelo de clasificación (RandomForest) entrenado "
        "con el dataset Pima Indians Diabetes para estimar si una paciente tiene diabetes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50,
            }
        }
    )

    pregnancies: int = Field(..., ge=0, le=20, description="Número de embarazos")
    glucose: float = Field(..., ge=0, le=300, description="Glucosa en plasma (mg/dL). 0 = no medido")
    blood_pressure: float = Field(..., ge=0, le=200, description="Presión diastólica (mm Hg). 0 = no medido")
    skin_thickness: float = Field(..., ge=0, le=100, description="Pliegue cutáneo del tríceps (mm). 0 = no medido")
    insulin: float = Field(..., ge=0, le=900, description="Insulina sérica a 2 h (mu U/ml). 0 = no medido")
    bmi: float = Field(..., ge=0, le=80, description="Índice de masa corporal. 0 = no medido")
    diabetes_pedigree_function: float = Field(..., ge=0, le=3, description="Función de antecedentes familiares de diabetes")
    age: int = Field(..., ge=1, le=120, description="Edad (años)")


class PredictResponse(BaseModel):
    request_id: str
    predicted_class: int
    predicted_label: str
    probability_diabetes: float
    latency_ms: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Registra cada solicitud entrante y su resultado (monitoreo básico)."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    _stats["total_requests"] += 1
    logger.info("[%s] %s %s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        _stats["errors_5xx"] += 1
        logger.exception("[%s] Error no controlado procesando la solicitud", request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno del servidor", "request_id": request_id},
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status_code >= 500:
        _stats["errors_5xx"] += 1
    elif response.status_code >= 400:
        _stats["errors_4xx"] += 1
    logger.info("[%s] status=%s tiempo=%.2fms", request_id, response.status_code, elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Registra como WARNING las solicitudes con datos inválidos y deja la respuesta 422 estándar."""
    errores = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    logger.warning("[%s] Entrada inválida -> %s", getattr(request.state, "request_id", "-"), errores)
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info():
    if not _metadata:
        raise HTTPException(status_code=503, detail="Metadatos del modelo no disponibles")
    return _metadata


@app.get("/stats")
def stats():
    """Contadores de monitoreo básico desde que arrancó el servicio."""
    n = _stats["predictions"]
    avg = _stats["predict_latency_ms_total"] / n if n else 0.0
    return {
        "started_at": _stats["started_at"],
        "total_requests": _stats["total_requests"],
        "errors_4xx": _stats["errors_4xx"],
        "errors_5xx": _stats["errors_5xx"],
        "predictions": n,
        "predictions_by_class": _stats["predictions_by_class"],
        "avg_predict_latency_ms": round(avg, 3),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    request_id = request.state.request_id
    start = time.perf_counter()

    if _model is None:
        logger.error("[%s] Solicitud de predicción con el modelo no cargado", request_id)
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        d = payload.model_dump()
        row = pd.DataFrame(
            [[
                d["pregnancies"], d["glucose"], d["blood_pressure"], d["skin_thickness"],
                d["insulin"], d["bmi"], d["diabetes_pedigree_function"], d["age"],
            ]],
            columns=FEATURES,
        )
        pred_class = int(_model.predict(row)[0])
        proba = float(_model.predict_proba(row)[0][1])
        label = _metadata["target_names"][pred_class]

        elapsed_ms = (time.perf_counter() - start) * 1000
        _stats["predictions"] += 1
        _stats["predictions_by_class"][label] += 1
        _stats["predict_latency_ms_total"] += elapsed_ms
        logger.info(
            "[%s] predicción OK: input=%s -> %s (p_diabetes=%.3f)",
            request_id, d, label, proba,
        )

        return PredictResponse(
            request_id=request_id,
            predicted_class=pred_class,
            predicted_label=label,
            probability_diabetes=round(proba, 4),
            latency_ms=round(elapsed_ms, 3),
        )
    except Exception:
        logger.exception("[%s] Error generando la predicción para input=%s", request_id, payload)
        raise HTTPException(status_code=500, detail="Error al generar la predicción")
