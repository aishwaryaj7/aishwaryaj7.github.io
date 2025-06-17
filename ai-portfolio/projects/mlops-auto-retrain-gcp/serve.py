#!/usr/bin/env python3
"""FastAPI serving script for churn prediction model."""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "churn_classifier"

# Global model variable
model = None
model_version = None

class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""
    SeniorCitizen: int = Field(..., description="Senior citizen status (0 or 1)")
    tenure: int = Field(..., ge=0, description="Number of months with company")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges amount")
    InternetService: str = Field(..., description="Internet service type")
    OnlineSecurity: str = Field(..., description="Online security service")
    TechSupport: str = Field(..., description="Tech support service")
    StreamingTV: str = Field(..., description="Streaming TV service")
    Contract: str = Field(..., description="Contract type")
    PaymentMethod: str = Field(..., description="Payment method")
    PaperlessBilling: str = Field(..., description="Paperless billing preference")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    customers: List[CustomerFeatures]

class PredictionResponse(BaseModel):
    """Prediction response."""
    customer_id: str = None
    churn_probability: float
    churn_prediction: str
    risk_level: str
    model_version: str
    prediction_timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    model_version: str

# FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Production-grade churn prediction service using MLflow models",
    version="1.0.0"
)

def load_model():
    """Load the latest model from MLflow."""
    global model, model_version

    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

        if not latest_versions:
            raise ValueError(f"No model versions found for {MODEL_NAME}")

        latest_version = latest_versions[0]
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"

        model = mlflow.sklearn.load_model(model_uri)
        model_version = latest_version.version

        logger.info(f"Loaded model {MODEL_NAME} version {model_version}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def get_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability."""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def prepare_features(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert customer features to DataFrame for prediction."""
    # Convert to dictionary and then DataFrame
    features_dict = customer.dict()

    # Convert SeniorCitizen to string format expected by model
    features_dict['SeniorCitizen'] = "Yes" if features_dict['SeniorCitizen'] == 1 else "No"

    # Convert TotalCharges to string (as expected by preprocessing)
    features_dict['TotalCharges'] = str(features_dict['TotalCharges'])

    return pd.DataFrame([features_dict])

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "model_version": model_version,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_type": str(type(model)),
        "features_expected": [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
            "InternetService", "OnlineSecurity", "TechSupport", "StreamingTV",
            "Contract", "PaymentMethod", "PaperlessBilling"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """Predict churn for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        features_df = prepare_features(customer)

        # Make prediction
        churn_prob = model.predict_proba(features_df)[0, 1]
        churn_pred = "Yes" if churn_prob >= 0.5 else "No"
        risk_level = get_risk_level(churn_prob)

        return PredictionResponse(
            churn_probability=float(churn_prob),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            model_version=str(model_version),
            prediction_timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_churn(request: BatchPredictionRequest):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []
        high_risk_count = 0

        for i, customer in enumerate(request.customers):
            # Prepare features
            features_df = prepare_features(customer)

            # Make prediction
            churn_prob = model.predict_proba(features_df)[0, 1]
            churn_pred = "Yes" if churn_prob >= 0.5 else "No"
            risk_level = get_risk_level(churn_prob)

            if risk_level == "HIGH":
                high_risk_count += 1

            predictions.append(PredictionResponse(
                customer_id=f"customer_{i+1}",
                churn_probability=float(churn_prob),
                churn_prediction=churn_pred,
                risk_level=risk_level,
                model_version=str(model_version),
                prediction_timestamp=datetime.now().isoformat()
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(request.customers),
            high_risk_count=high_risk_count,
            model_version=str(model_version)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )