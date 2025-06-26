"""
Pydantic schemas for Energy Price Forecasting API.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class CountryCode(str, Enum):
    """Supported country codes."""
    DE = "DE"  # Germany
    FR = "FR"  # France
    NL = "NL"  # Netherlands

class ModelType(str, Enum):
    """Available model types."""
    NAIVE = "naive"
    ARIMA = "arima"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"

class ForecastRequest(BaseModel):
    """Request schema for price forecasting."""
    country: CountryCode = Field(..., description="Country code for forecasting")
    model_name: str = Field(..., description="Name of the model to use")
    forecast_horizon: int = Field(24, ge=1, le=168, description="Number of hours to forecast (1-168)")
    start_datetime: Optional[datetime] = Field(None, description="Start datetime for forecast (defaults to now)")
    include_confidence_intervals: bool = Field(True, description="Whether to include confidence intervals")
    
    class Config:
        json_schema_extra = {
            "example": {
                "country": "DE",
                "model_name": "xgboost",
                "forecast_horizon": 24,
                "start_datetime": "2024-01-15T00:00:00",
                "include_confidence_intervals": True
            }
        }

class PredictionPoint(BaseModel):
    """Single prediction point."""
    prediction_datetime: datetime = Field(..., description="Datetime for the prediction", alias="datetime")
    predicted_price: float = Field(..., description="Predicted price in EUR/MWh")
    confidence_lower: Optional[float] = Field(None, description="Lower confidence bound")
    confidence_upper: Optional[float] = Field(None, description="Upper confidence bound")
    country: CountryCode = Field(..., description="Country code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "datetime": "2024-01-15T00:00:00",
                "predicted_price": 45.67,
                "confidence_lower": 41.10,
                "confidence_upper": 50.24,
                "country": "DE"
            }
        }

class ForecastResponse(BaseModel):
    """Response schema for price forecasting."""
    predictions: List[PredictionPoint] = Field(..., description="List of price predictions")
    model_used: str = Field(..., description="Name of the model used")
    forecast_horizon: int = Field(..., description="Number of hours forecasted")
    country: CountryCode = Field(..., description="Country code")
    generated_at: datetime = Field(..., description="Timestamp when forecast was generated")
    model_version: Optional[datetime] = Field(None, description="Model training timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "datetime": "2024-01-15T00:00:00",
                        "predicted_price": 45.67,
                        "confidence_lower": 41.10,
                        "confidence_upper": 50.24,
                        "country": "DE"
                    }
                ],
                "model_used": "xgboost",
                "forecast_horizon": 24,
                "country": "DE",
                "generated_at": "2024-01-15T12:00:00",
                "model_version": "2024-01-14T10:30:00"
            }
        }

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Energy Price Forecasting API is running",
                "timestamp": "2024-01-15T12:00:00",
                "version": "1.0.0"
            }
        }

class ModelInfo(BaseModel):
    """Model information schema."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    is_fitted: bool = Field(..., description="Whether model is fitted")
    training_timestamp: Optional[datetime] = Field(None, description="When model was trained")
    feature_count: int = Field(..., description="Number of features used")
    target_column: str = Field(..., description="Target column name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "xgboost",
                "type": "xgboost",
                "is_fitted": True,
                "training_timestamp": "2024-01-14T10:30:00",
                "feature_count": 45,
                "target_column": "price"
            }
        }

class ModelPerformance(BaseModel):
    """Model performance metrics schema."""
    model_name: str = Field(..., description="Model name")
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R-squared score")
    directional_accuracy: float = Field(..., description="Directional accuracy percentage")
    last_evaluated: datetime = Field(..., description="Last evaluation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "xgboost",
                "mae": 8.45,
                "rmse": 12.67,
                "mape": 15.23,
                "r2": 0.87,
                "directional_accuracy": 72.5,
                "last_evaluated": "2024-01-14T15:30:00"
            }
        }

class ArbitrageOpportunity(BaseModel):
    """Arbitrage opportunity schema."""
    opportunity_datetime: datetime = Field(..., description="Datetime of opportunity", alias="datetime")
    buy_country: CountryCode = Field(..., description="Country to buy from")
    sell_country: CountryCode = Field(..., description="Country to sell to")
    buy_price: float = Field(..., description="Buy price in EUR/MWh")
    sell_price: float = Field(..., description="Sell price in EUR/MWh")
    spread: float = Field(..., description="Price spread in EUR/MWh")
    spread_percentage: float = Field(..., description="Spread as percentage")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "datetime": "2024-01-15T14:00:00",
                "buy_country": "NL",
                "sell_country": "DE",
                "buy_price": 42.50,
                "sell_price": 48.75,
                "spread": 6.25,
                "spread_percentage": 14.7,
                "confidence_score": 0.85
            }
        }

class ArbitrageRequest(BaseModel):
    """Request schema for arbitrage opportunity detection."""
    countries: List[CountryCode] = Field(..., description="Countries to analyze")
    forecast_horizon: int = Field(24, ge=1, le=168, description="Forecast horizon in hours")
    min_spread_threshold: float = Field(5.0, ge=0, description="Minimum spread threshold in EUR/MWh")
    min_confidence: float = Field(0.7, ge=0, le=1, description="Minimum confidence threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "countries": ["DE", "FR", "NL"],
                "forecast_horizon": 24,
                "min_spread_threshold": 5.0,
                "min_confidence": 0.7
            }
        }

class ArbitrageResponse(BaseModel):
    """Response schema for arbitrage opportunities."""
    opportunities: List[ArbitrageOpportunity] = Field(..., description="List of arbitrage opportunities")
    total_opportunities: int = Field(..., description="Total number of opportunities found")
    analysis_period: Dict[str, datetime] = Field(..., description="Analysis period start and end")
    generated_at: datetime = Field(..., description="Analysis timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "opportunities": [
                    {
                        "datetime": "2024-01-15T14:00:00",
                        "buy_country": "NL",
                        "sell_country": "DE",
                        "buy_price": 42.50,
                        "sell_price": 48.75,
                        "spread": 6.25,
                        "spread_percentage": 14.7,
                        "confidence_score": 0.85
                    }
                ],
                "total_opportunities": 1,
                "analysis_period": {
                    "start": "2024-01-15T00:00:00",
                    "end": "2024-01-16T00:00:00"
                },
                "generated_at": "2024-01-15T12:00:00"
            }
        }

class RetrainingRequest(BaseModel):
    """Request schema for model retraining."""
    force_retrain: bool = Field(False, description="Force retraining even if recent model exists")
    data_start_date: Optional[datetime] = Field(None, description="Start date for training data")
    data_end_date: Optional[datetime] = Field(None, description="End date for training data")
    models_to_retrain: Optional[List[str]] = Field(None, description="Specific models to retrain")
    
    class Config:
        json_schema_extra = {
            "example": {
                "force_retrain": False,
                "data_start_date": "2024-01-01T00:00:00",
                "data_end_date": "2024-01-14T23:59:59",
                "models_to_retrain": ["xgboost", "lightgbm"]
            }
        }

class RetrainingResponse(BaseModel):
    """Response schema for retraining requests."""
    status: str = Field(..., description="Retraining status")
    message: str = Field(..., description="Status message")
    job_id: Optional[str] = Field(None, description="Background job ID")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    started_at: datetime = Field(..., description="When retraining started")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "started",
                "message": "Model retraining job started successfully",
                "job_id": "retrain_20240115_120000",
                "estimated_completion": "2024-01-15T14:00:00",
                "started_at": "2024-01-15T12:00:00"
            }
        }
