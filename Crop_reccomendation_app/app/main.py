from __future__ import annotations

import os
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = FastAPI(
    title="Crop Recommendation API",
    version=os.getenv("APP_VERSION", "1.0.0"),
    summary="Predict the best crop based on soil and weather features",
)

class CropFeatures(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., description="Relative humidity (%)")
    ph: float = Field(..., description="Soil pH value")
    rainfall: float = Field(..., description="Rainfall (mm)")


model = None
label_encoder = None


def load_artifacts() -> None:
    global model, label_encoder
    model_path = os.getenv("MODEL_PATH", "model_artifacts/model.joblib")
    le_path = os.getenv("LE_PATH", "model_artifacts/le.joblib")

    if not os.path.exists(model_path) or not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Model artifacts not found. Ensure '{model_path}' and '{le_path}' exist."
        )

    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: CropFeatures) -> Dict[str, Any]:
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    df = pd.DataFrame(
        [
            [
                features.N,
                features.P,
                features.K,
                features.temperature,
                features.humidity,
                features.ph,
                features.rainfall,
            ]
        ],
        columns=FEATURE_NAMES,
    )

    try:
        encoded_pred = int(model.predict(df)[0])
        crop_name = label_encoder.inverse_transform([encoded_pred])[0]

        response: Dict[str, Any] = {
            "prediction": crop_name,
            "encoded_label": encoded_pred,
        }

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            top_idx = np.argsort(proba)[::-1][:3]
            response["top_3"] = [
                {
                    "crop": label_encoder.inverse_transform([int(i)])[0],
                    "probability": float(proba[i]),
                }
                for i in top_idx
            ]

        return response
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)

