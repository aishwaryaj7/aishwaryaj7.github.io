"""
Utility functions for Energy Price Forecasting project.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pytz

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('energy_forecasting.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        "total_rows": len(df),
        "missing_columns": [],
        "missing_values": {},
        "duplicate_rows": 0,
        "data_types": {},
        "date_range": {},
        "quality_score": 0.0
    }
    
    # Check for missing columns
    quality_report["missing_columns"] = [col for col in required_columns if col not in df.columns]
    
    # Check for missing values
    quality_report["missing_values"] = df.isnull().sum().to_dict()
    
    # Check for duplicates
    quality_report["duplicate_rows"] = df.duplicated().sum()
    
    # Data types
    quality_report["data_types"] = df.dtypes.to_dict()
    
    # Date range (if datetime column exists)
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        date_col = datetime_cols[0]
        quality_report["date_range"] = {
            "start": df[date_col].min(),
            "end": df[date_col].max(),
            "total_hours": (df[date_col].max() - df[date_col].min()).total_seconds() / 3600
        }
    
    # Calculate quality score (0-1)
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    quality_report["quality_score"] = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
    
    return quality_report

def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Create time-based features from datetime column.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    
    # Basic time features
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_month'] = dt.dt.day
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['year'] = dt.dt.year
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Business day indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_day'] = (~pd.to_datetime(df[datetime_col]).dt.date.isin(
        pd.bdate_range(dt.min(), dt.max()).date
    )).astype(int)
    
    return df

def create_lag_features(df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series.
    
    Args:
        df: Input DataFrame (should be sorted by time)
        target_col: Name of target column
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    
    return df

def detect_outliers(series: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        series: Input time series
        method: Method to use ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")

def get_market_hours() -> Dict[str, List[int]]:
    """
    Get typical market trading hours for different regions.
    
    Returns:
        Dictionary mapping regions to trading hours
    """
    return {
        "DE": list(range(24)),  # 24/7 trading
        "FR": list(range(24)),  # 24/7 trading
        "NL": list(range(24)),  # 24/7 trading
        "peak_hours": [8, 9, 10, 11, 18, 19, 20, 21],  # Typical peak hours
        "off_peak_hours": [0, 1, 2, 3, 4, 5, 22, 23]   # Typical off-peak hours
    }

def convert_timezone(df: pd.DataFrame, datetime_col: str, 
                    from_tz: str = "UTC", to_tz: str = "Europe/Berlin") -> pd.DataFrame:
    """
    Convert timezone of datetime column.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of datetime column
        from_tz: Source timezone
        to_tz: Target timezone
        
    Returns:
        DataFrame with converted timezone
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    if df[datetime_col].dt.tz is None:
        df[datetime_col] = df[datetime_col].dt.tz_localize(from_tz)
    
    df[datetime_col] = df[datetime_col].dt.tz_convert(to_tz)
    
    return df
