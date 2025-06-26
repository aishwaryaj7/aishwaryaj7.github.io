"""
FastAPI application for Energy Price Forecasting service.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
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
    ModelInfo, PredictionPoint, ModelPerformance
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
        
        # Load recent data for feature engineering
        end_date = request.start_datetime or datetime.now()
        start_date = end_date - timedelta(days=7)  # Get last week of data for features
        
        # For demo purposes, create synthetic recent data
        # In production, this would load actual recent data
        recent_data = create_synthetic_recent_data(start_date, end_date, request.country)
        
        # Generate features for prediction
        feature_data = prepare_prediction_features(
            recent_data, 
            request.forecast_horizon,
            model_artifacts.get('feature_columns', [])
        )
        
        # Generate predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(feature_data)
        else:
            # Fallback for baseline models
            predictions = np.random.normal(50, 10, request.forecast_horizon)  # Demo predictions
        
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

@app.get("/models/{model_name}/performance", response_model=ModelPerformance)
async def get_model_performance(model_name: str):
    """
    Get performance metrics for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model performance metrics
    """
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    models = model_artifacts.get('models', {})
    
    if model_name not in models:
        available_models = list(models.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {available_models}"
        )
    
    # For demo purposes, return synthetic performance metrics
    # In production, these would be loaded from evaluation results
    performance = ModelPerformance(
        model_name=model_name,
        mae=np.random.uniform(5, 15),
        rmse=np.random.uniform(8, 20),
        mape=np.random.uniform(10, 25),
        r2=np.random.uniform(0.7, 0.95),
        directional_accuracy=np.random.uniform(60, 85),
        last_evaluated=datetime.now() - timedelta(days=1)
    )
    
    return performance

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

def create_synthetic_recent_data(start_date: datetime, end_date: datetime, country: str) -> pd.DataFrame:
    """
    Create synthetic recent data for demo purposes.
    In production, this would load actual recent data.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create synthetic price data with realistic patterns
    hours = np.array([dt.hour for dt in date_range])
    days = np.array([dt.dayofweek for dt in date_range])
    
    # Base price with hourly and daily patterns
    base_price = 50
    hourly_pattern = 10 * np.sin(2 * np.pi * hours / 24)
    daily_pattern = 5 * np.sin(2 * np.pi * days / 7)
    noise = np.random.normal(0, 5, len(date_range))
    
    prices = base_price + hourly_pattern + daily_pattern + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': date_range,
        'price': prices,
        'country': country,
        'load': np.random.normal(30000, 5000, len(date_range)),
        'renewable_generation': np.random.normal(10000, 3000, len(date_range)),
        'temperature': np.random.normal(15, 8, len(date_range)),
        'wind_speed': np.random.normal(8, 4, len(date_range))
    })
    
    return data

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
    # For demo purposes, create a simple feature matrix
    # In production, this would use the same feature engineering as training
    
    n_features = len(feature_columns) if feature_columns else 10
    
    # Create synthetic feature matrix
    feature_data = pd.DataFrame(
        np.random.randn(forecast_horizon, n_features),
        columns=feature_columns[:n_features] if feature_columns else [f'feature_{i}' for i in range(n_features)]
    )
    
    return feature_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
