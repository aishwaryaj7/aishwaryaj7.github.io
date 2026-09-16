"""
FastAPI application for Energy Price Forecasting service.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.loaders import GCSDataLoader
from src.utils.config import config
from src.utils.helpers import setup_logging
from .schemas import (
    ForecastRequest, ForecastResponse, HealthResponse,
    ModelInfo, PredictionPoint
)

logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Energy Price Forecasting API",
    description="API for predicting intraday electricity prices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data loader
model_artifacts = None
data_loader = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global model_artifacts, data_loader
    
    logger.info("Starting Energy Price Forecasting API")
    
    # Initialize data loader
    data_loader = GCSDataLoader()
    
    # Load latest model artifacts
    try:
        model_artifacts = data_loader.load_model_artifacts("energy_forecasting", "latest")
        if model_artifacts:
            logger.info("Model artifacts loaded successfully")
        else:
            logger.warning("No model artifacts found. Some endpoints may not work.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        model_artifacts = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information."""
    return HealthResponse(
        status="healthy",
        message="Energy Price Forecasting API is running",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model_artifacts else "not_loaded"
    
    return HealthResponse(
        status="healthy",
        message=f"API is healthy. Model status: {model_status}",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/models/info", response_model=List[ModelInfo])
async def get_model_info():
    """Get information about available models."""
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    models_info = []
    
    if 'models' in model_artifacts:
        for model_name, model in model_artifacts['models'].items():
            model_info = ModelInfo(
                name=model_name,
                type=getattr(model, 'model_type', 'unknown'),
                is_fitted=getattr(model, 'is_fitted', False),
                training_timestamp=model_artifacts.get('training_timestamp'),
                feature_count=len(model_artifacts.get('feature_columns', [])),
                target_column=model_artifacts.get('target_column', 'price')
            )
            models_info.append(model_info)
    
    return models_info

@app.post("/predict", response_model=ForecastResponse)
async def predict_prices(request: ForecastRequest):
    """
    Generate price forecasts.
    
    Args:
        request: Forecast request with parameters
        
    Returns:
        Forecast response with predictions
    """
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    try:
        # Get the requested model
        models = model_artifacts.get('models', {})
        
        if request.model_name not in models:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model_name}' not found. Available models: {available_models}"
            )
        
        model = models[request.model_name]
        
        # Load the most recent processed data produced by the training pipeline.
        recent_data = data_loader.load_latest_processed_data(processing_stage="final")
        if recent_data is None or len(recent_data) == 0:
            raise HTTPException(
                status_code=503,
                detail="No processed data available. Run the data pipeline before requesting a forecast."
            )
        
        # Generate features for prediction
        feature_data = prepare_prediction_features(
            recent_data, 
            request.forecast_horizon,
            model_artifacts.get('feature_columns', [])
        )
        
        # Generate predictions
        if not hasattr(model, "predict"):
            raise HTTPException(
                status_code=500,
                detail=f"Model '{request.model_name}' exposes no predict method."
            )
        predictions = model.predict(feature_data)
        
        # Create prediction points
        prediction_points = []
        current_time = request.start_datetime or datetime.now()
        
        for i, pred in enumerate(predictions[:request.forecast_horizon]):
            point = PredictionPoint(
                prediction_datetime=current_time + timedelta(hours=i),
                predicted_price=float(pred),
                confidence_lower=float(pred * 0.9),  # Simplified confidence intervals
                confidence_upper=float(pred * 1.1),
                country=request.country
            )
            prediction_points.append(point)
        
        response = ForecastResponse(
            predictions=prediction_points,
            model_used=request.model_name,
            forecast_horizon=request.forecast_horizon,
            country=request.country,
            generated_at=datetime.now(),
            model_version=model_artifacts.get('training_timestamp')
        )
        
        logger.info(f"Generated {len(prediction_points)} predictions using {request.model_name}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background.
    
    Returns:
        Status message
    """
    background_tasks.add_task(retrain_models)
    
    return {
        "message": "Model retraining triggered",
        "status": "started",
        "timestamp": datetime.now()
    }

async def retrain_models():
    """Background task for model retraining."""
    try:
        logger.info("Starting background model retraining")
        
        # Import training pipeline
        from train import EnergyForecastingPipeline
        
        # Run training pipeline
        pipeline = EnergyForecastingPipeline()
        await pipeline.run_full_pipeline()
        
        # Reload model artifacts
        global model_artifacts
        model_artifacts = data_loader.load_model_artifacts("energy_forecasting", "latest")
        
        logger.info("Background model retraining completed")
        
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")

def prepare_prediction_features(data: pd.DataFrame, forecast_horizon: int, 
                              feature_columns: List[str]) -> pd.DataFrame:
    """
    Prepare features for prediction.
    
    Args:
        data: Recent historical data
        forecast_horizon: Number of hours to forecast
        feature_columns: List of required feature columns
        
    Returns:
        DataFrame with features for prediction
    """
    missing = [c for c in feature_columns if c not in data.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Processed data is missing feature columns the model needs: {missing}"
        )

    # Use the most recent rows as the basis for the forecast horizon.
    features = data[feature_columns].tail(forecast_horizon)
    if len(features) < forecast_horizon:
        raise HTTPException(
            status_code=503,
            detail=f"Only {len(features)} rows of processed data available, "
                   f"need {forecast_horizon} for this horizon."
        )
    return features.reset_index(drop=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
