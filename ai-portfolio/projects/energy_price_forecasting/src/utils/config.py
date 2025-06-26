"""
Configuration management for Energy Price Forecasting project.
"""
import os
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class APIConfig:
    """API configuration settings."""
    entso_e_token: str = os.getenv("ENTSO_E_TOKEN", "")
    openweather_token: str = os.getenv("OPENWEATHER_API_KEY", "")
    base_urls: Dict[str, str] = None
    
    def __post_init__(self):
        if self.base_urls is None:
            self.base_urls = {
                "entso_e": "https://web-api.tp.entsoe.eu/api",
                "openweather": "https://api.openweathermap.org/data/2.5"
            }

@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    bucket_name: str = os.getenv("GCS_BUCKET_NAME", "energy-forecasting-data")
    credentials_path: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    region: str = os.getenv("GCP_REGION", "us-central1")

@dataclass
class DataConfig:
    """Data processing configuration."""
    # Market zones for EPEX Spot
    market_zones: List[str] = None
    # Time series parameters
    forecast_horizon: int = 24  # hours
    lookback_window: int = 168  # 7 days in hours
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    def __post_init__(self):
        if self.market_zones is None:
            self.market_zones = ["DE", "FR", "NL"]  # Germany, France, Netherlands

@dataclass
class ModelConfig:
    """Model training configuration."""
    # Cross-validation parameters
    cv_folds: int = 5
    test_size: float = 0.2
    # Model parameters
    random_state: int = 42
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mae", "rmse", "mape", "pinball_loss"]

@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = "energy_price_forecasting"
    artifact_location: str = os.getenv("MLFLOW_ARTIFACT_LOCATION", "./mlruns")

@dataclass
class Config:
    """Main configuration class."""
    api: APIConfig = None
    gcp: GCPConfig = None
    data: DataConfig = None
    model: ModelConfig = None
    mlflow: MLflowConfig = None
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    
    def __post_init__(self):
        if self.api is None:
            self.api = APIConfig()
        if self.gcp is None:
            self.gcp = GCPConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.mlflow is None:
            self.mlflow = MLflowConfig()

# Global configuration instance
config = Config()
