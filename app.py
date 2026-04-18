import uuid
import os
import logging
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model state - loaded once at startup, reused for every request

MODEL_STATE: dict = {}

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/rf_baseline.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
MODEL_NAME   = "taxi-tip-regressor"
MODEL_VERSION = "1"

# Training metrics recorded during MLflow experiment (RF_Baseline)
TRAINING_METRICS = {"MAE": 1.1823, "RMSE": 2.2675, "R2": 0.6437}

FEATURES = [
    "pickup_hour", "pickup_day_of_week", "is_weekend",
    "trip_distance", "trip_duration_minutes", "trip_speed_mph", "log_trip_distance",
    "fare_amount", "fare_per_mile", "fare_per_minute",
    "passenger_count",
    "pickup_borough_encoded", "dropoff_borough_encoded",
    "tolls_amount", "extra", "mta_tax", "congestion_surcharge", "Airport_fee",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler once when the application starts."""
    logger.info("Loading model and scaler...")
    MODEL_STATE["model"]   = joblib.load(MODEL_PATH)
    MODEL_STATE["scaler"]  = joblib.load(SCALER_PATH)
    MODEL_STATE["loaded"]  = True
    logger.info("Model loaded successfully.")
    yield
    MODEL_STATE.clear()



# App

app = FastAPI(
    title="Taxi Tip Prediction API",
    description="Predicts NYC Yellow Taxi tip amounts using a Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)



# Global exception handler - returns structured JSON, hides internal details

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )



# Pydantic schemas

class TripFeatures(BaseModel):
    """Input schema for a single trip prediction request.

    All 18 features that the model was trained on are required.
    At least 5 fields carry explicit validation constraints.
    """
    # Time features
    pickup_hour:        int   = Field(..., ge=0, le=23,
                                      description="Hour of pickup (0-23)")
    pickup_day_of_week: int   = Field(..., ge=0, le=6,
                                      description="Day of week (0=Monday, 6=Sunday)")
    is_weekend:         int   = Field(..., ge=0, le=1,
                                      description="1 if weekend, 0 otherwise")

    # Distance / duration features
    trip_distance:          float = Field(..., gt=0,
                                          description="Trip distance in miles (must be positive)")
    trip_duration_minutes:  float = Field(..., gt=0,
                                          description="Trip duration in minutes (must be positive)")
    trip_speed_mph:         float = Field(..., ge=0,
                                          description="Average speed in mph")
    log_trip_distance:      float = Field(...,
                                          description="Natural log of trip_distance")

    # Fare features
    fare_amount:    float = Field(..., gt=0,
                                  description="Base fare in USD (must be positive)")
    fare_per_mile:  float = Field(..., ge=0, description="Fare per mile")
    fare_per_minute: float = Field(..., ge=0, description="Fare per minute")

    # Passenger
    passenger_count: int = Field(..., ge=1, le=8,
                                  description="Number of passengers (1-8)")

    # Borough encodings
    pickup_borough_encoded:  int = Field(..., ge=0, le=5,
                                          description="Encoded pickup borough (0-5)")
    dropoff_borough_encoded: int = Field(..., ge=0, le=5,
                                          description="Encoded dropoff borough (0-5)")

    # Surcharges / extras
    tolls_amount:        float = Field(..., ge=0, description="Tolls in USD")
    extra:               float = Field(..., ge=0, description="Extra charges in USD")
    mta_tax:             float = Field(..., ge=0, description="MTA tax in USD")
    congestion_surcharge: float = Field(..., ge=0, description="Congestion surcharge in USD")
    Airport_fee:         float = Field(..., ge=0, description="Airport fee in USD")

    model_config = {"json_schema_extra": {"example": {
        "pickup_hour": 14, "pickup_day_of_week": 2, "is_weekend": 0,
        "trip_distance": 3.5, "trip_duration_minutes": 18.0, "trip_speed_mph": 11.7,
        "log_trip_distance": 1.45, "fare_amount": 16.5, "fare_per_mile": 4.71,
        "fare_per_minute": 0.92, "passenger_count": 1,
        "pickup_borough_encoded": 3, "dropoff_borough_encoded": 3,
        "tolls_amount": 0.0, "extra": 0.5, "mta_tax": 0.5,
        "congestion_surcharge": 2.5, "Airport_fee": 0.0,
    }}}


class BatchRequest(BaseModel):
    """Up to 100 trip records for batch prediction."""
    trips: List[TripFeatures] = Field(..., min_length=1, max_length=100,
                                       description="List of trip records (1-100)")


class PredictionResponse(BaseModel):
    prediction_id: str
    tip_amount:    float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions:   List[PredictionResponse]
    model_version: str
    count:         int



# Helper

def _predict_one(trip: TripFeatures) -> float:
    """Scale features and return a raw model prediction."""
    row = np.array([[getattr(trip, f) for f in FEATURES]])
    scaled = MODEL_STATE["scaler"].transform(row)
    return float(MODEL_STATE["model"].predict(scaled)[0])



# Endpoints

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(trip: TripFeatures):
    """Return a predicted tip amount for a single trip."""
    tip = round(_predict_one(trip), 2)
    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        tip_amount=tip,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    """Return predicted tip amounts for up to 100 trips."""
    predictions = [
        PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            tip_amount=round(_predict_one(trip), 2),
            model_version=MODEL_VERSION,
        )
        for trip in batch.trips
    ]
    return BatchPredictionResponse(
        predictions=predictions,
        model_version=MODEL_VERSION,
        count=len(predictions),
    )


@app.get("/health", tags=["Operations"])
def health():
    """Return API and model status."""
    return {
        "status":        "ok",
        "model_loaded":  MODEL_STATE.get("loaded", False),
        "model_version": MODEL_VERSION,
    }


@app.get("/model/info", tags=["Operations"])
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name":       MODEL_NAME,
        "model_version":    MODEL_VERSION,
        "feature_names":    FEATURES,
        "num_features":     len(FEATURES),
        "training_metrics": TRAINING_METRICS,
    }
