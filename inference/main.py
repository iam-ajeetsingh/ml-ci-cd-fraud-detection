import os
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.model_loader import get_production_info, load_production_model, RegistryError


app = FastAPI(title="Fraud Model Inference API", version="0.1")

DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))


class PredictRequest(BaseModel):
    # creditcard.csv has 30 features: Time, V1..V28, Amount
    # We'll accept a list in exactly that order.
    features: List[float] = Field(..., min_length=30, max_length=30)


class PredictResponse(BaseModel):
    model_version: str
    fraud_probability: float
    fraud_label: int
    threshold: float


@app.get("/health")
def health():
    try:
        info = get_production_info()
        return {"status": "ok", **info}
    except RegistryError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model, registry = load_production_model()
    except RegistryError as e:
        raise HTTPException(status_code=500, detail=str(e))

    x = np.array(req.features, dtype=float).reshape(1, -1)

    # LogisticRegression supports predict_proba
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not support predict_proba")

    proba = float(model.predict_proba(x)[0, 1])

    threshold = DEFAULT_THRESHOLD
    label = 1 if proba >= threshold else 0

    return PredictResponse(
        model_version=registry.get("current_model", "unknown"),
        fraud_probability=proba,
        fraud_label=label,
        threshold=threshold,
    )
