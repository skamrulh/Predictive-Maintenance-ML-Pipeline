"""
FastAPI application for predictive maintenance system.
Provides REST API endpoints for failure probability and RUL predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import yaml
import pandas as pd
import os
from typing import Dict, Any


# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting equipment failures and remaining useful life",
    version="1.0.0"
)


# Load configuration
try:
    # FILENAME CORRECTION: Use config.yaml
    with open("config.yaml", "r") as f:
        CFG: Dict[str, Any] = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError("Configuration file 'config.yaml' not found")


# Load trained models
try:
    CLF_MODEL = joblib.load(CFG["paths"]["model_cls"])
    RUL_MODEL = joblib.load(CFG["paths"]["model_rul"])
    print("✅ Models loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Model files not found: {e}")
    print("Please train the models first using: 'python train_classification.py' and 'python train_rul.py'")
    CLF_MODEL = None
    RUL_MODEL = None


class EquipmentSnapshot(BaseModel):
    """Pydantic model for equipment sensor data."""
    temp: float
    vibration: float
    pressure: float
    current: float
    sound: float


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    message: str
    data: dict


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_loaded = CLF_MODEL is not None and RUL_MODEL is not None
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "classification_model": "loaded" if CLF_MODEL else "missing",
        "rul_model": "loaded" if RUL_MODEL else "missing"
    }


@app.post("/predict/failure_proba", response_model=PredictionResponse)
def predict_failure_probability(snapshot: EquipmentSnapshot):
    """Predict probability of equipment failure within 24 hours."""
    if CLF_MODEL is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    
    try:
        # Pydantic V2 fix: use model_dump() instead of dict()
        raw_data = snapshot.model_dump()
        
        # CRITICAL FIX: The classification model was retrained on 5 mean features.
        # We must create those feature columns from the raw input for the API call.
        # Since this is a single snapshot, we treat the raw value as the 'mean'.
        input_row: Dict[str, float] = {f"{k}_mean": v for k, v in raw_data.items()}
        input_data = pd.DataFrame([input_row])
        
        # Get failure probability
        failure_proba = CLF_MODEL.predict_proba(input_data)[:, 1][0]
        
        return PredictionResponse(
            success=True,
            message="Failure probability prediction successful",
            data={"fail_within_24h_proba": float(failure_proba)}
        )
    except Exception as e:
        # Now this error will only fire if the pipeline itself fails
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/rul", response_model=PredictionResponse)
def predict_rul(snapshot: EquipmentSnapshot):
    """Predict Remaining Useful Life (RUL) of equipment in hours."""
    if RUL_MODEL is None:
        raise HTTPException(status_code=503, detail="RUL model not loaded")
    
    try:
        # RUL model uses raw features, so we just convert the snapshot
        input_data = pd.DataFrame([snapshot.model_dump()]) # Use model_dump()
        
        # Predict RUL
        rul_prediction = RUL_MODEL.predict(input_data)[0]
        
        rul_prediction = max(0.0, rul_prediction)
        
        return PredictionResponse(
            success=True,
            message="RUL prediction successful",
            data={"predicted_RUL_hours": float(rul_prediction)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RUL prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")