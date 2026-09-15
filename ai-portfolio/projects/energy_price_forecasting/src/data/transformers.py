"""
Data transformation module for Energy Price Forecasting project.
Handles data cleaning, feature engineering, and preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from ..utils.helpers import (
    create_time_features, create_lag_features, create_rolling_features,
    detect_outliers, validate_data_quality, setup_logging
)

logger = setup_logging()

class EnergyDataTransformer:
    """
    Main transformer class for energy price forecasting data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        self.target_column = 'price'
        
    def transform_raw_data(self, price_data: pd.DataFrame, load_data: pd.DataFrame,
                          renewable_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and combine all raw data sources into a unified dataset.
        
        Args:
            price_data: Day-ahead price data
            load_data: Electricity load data
            renewable_data: Renewable generation data
            weather_data: Weather data
            
        Returns:
            Transformed and combined DataFrame
        """
        logger.info("Starting data transformation process")
        
        # Validate input data
        self._validate_input_data(price_data, load_data, renewable_data, weather_data)
        
        # Clean individual datasets
        price_clean = self._clean_price_data(price_data)
        load_clean = self._clean_load_data(load_data)
        renewable_clean = self._clean_renewable_data(renewable_data)
        weather_clean = self._clean_weather_data(weather_data)
        
        # Merge all datasets
        combined_data = self._merge_datasets(price_clean, load_clean, renewable_clean, weather_clean)
        
        # Create features
        featured_data = self._create_features(combined_data)
        
        # Handle missing values
        final_data = self._handle_missing_values(featured_data)
        
        logger.info(f"Data transformation completed. Final dataset shape: {final_data.shape}")
        return final_data
    
    def _validate_input_data(self, price_data: pd.DataFrame, load_data: pd.DataFrame,
                           renewable_data: pd.DataFrame, weather_data: pd.DataFrame):
        """Validate input data quality."""
        datasets = {
            'price': price_data,
            'load': load_data,
            'renewable': renewable_data,
            'weather': weather_data
        }
        
        for name, df in datasets.items():
            if df.empty:
                logger.warning(f"{name} dataset is empty")
                continue
                
            quality_report = validate_data_quality(df, ['datetime', 'country'])
            logger.info(f"{name} data quality score: {quality_report['quality_score']:.2f}")
            
            if quality_report['quality_score'] < 0.8:
                logger.warning(f"{name} dataset has low quality score: {quality_report['quality_score']:.2f}")
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize price data."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Ensure datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Remove negative prices (can happen in some markets but unusual)
        negative_prices = df['price'] < 0
        if negative_prices.sum() > 0:
            logger.info(f"Found {negative_prices.sum()} negative prices, keeping them as they can be valid")
        
        # Detect and handle extreme outliers (>1000 EUR/MWh or <-500 EUR/MWh)
        extreme_outliers = (df['price'] > 1000) | (df['price'] < -500)
        if extreme_outliers.sum() > 0:
            logger.warning(f"Found {extreme_outliers.sum()} extreme price outliers")
            df.loc[extreme_outliers, 'price'] = np.nan
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def _clean_load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize load data."""
        if df.empty:
            return df
            
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Remove negative load values
        negative_load = df['load'] < 0
        if negative_load.sum() > 0:
            logger.warning(f"Found {negative_load.sum()} negative load values, setting to NaN")
            df.loc[negative_load, 'load'] = np.nan
        
        # Detect unrealistic load values (>100,000 MW for these countries)
        extreme_load = df['load'] > 100000
        if extreme_load.sum() > 0:
            logger.warning(f"Found {extreme_load.sum()} extreme load values")
            df.loc[extreme_load, 'load'] = np.nan
        
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    
    def _clean_renewable_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize renewable generation data."""
        if df.empty:
            return df
            
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Handle different column names from the extractor
        if 'renewable_total' in df.columns:
            df['renewable_generation'] = df['renewable_total']
        elif 'generation' in df.columns:
            df['renewable_generation'] = df['generation']
        
        # Remove negative generation values
        if 'renewable_generation' in df.columns:
            negative_gen = df['renewable_generation'] < 0
            if negative_gen.sum() > 0:
                logger.warning(f"Found {negative_gen.sum()} negative renewable generation values")
                df.loc[negative_gen, 'renewable_generation'] = 0
        
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    
    def _clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize weather data."""
        if df.empty:
            return df
            
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Validate temperature ranges (-50 to 50 Celsius)
        if 'temperature' in df.columns:
            extreme_temp = (df['temperature'] < -50) | (df['temperature'] > 50)
            if extreme_temp.sum() > 0:
                logger.warning(f"Found {extreme_temp.sum()} extreme temperature values")
                df.loc[extreme_temp, 'temperature'] = np.nan
        
        # Validate wind speed (0 to 200 km/h)
        if 'wind_speed' in df.columns:
            extreme_wind = (df['wind_speed'] < 0) | (df['wind_speed'] > 200)
            if extreme_wind.sum() > 0:
                logger.warning(f"Found {extreme_wind.sum()} extreme wind speed values")
                df.loc[extreme_wind, 'wind_speed'] = np.nan
        
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    
    def _merge_datasets(self, price_data: pd.DataFrame, load_data: pd.DataFrame,
                       renewable_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on datetime and country."""
        # Start with price data as the base
        if price_data.empty:
            logger.error("Price data is empty, cannot proceed with merging")
            return pd.DataFrame()
        
        merged = price_data.copy()
        
        # Merge load data
        if not load_data.empty:
            merged = pd.merge(merged, load_data, on=['datetime', 'country'], how='left')
            logger.info(f"Merged load data. Shape: {merged.shape}")
        
        # Merge renewable data
        if not renewable_data.empty:
            renewable_cols = ['datetime', 'country']
            if 'renewable_generation' in renewable_data.columns:
                renewable_cols.append('renewable_generation')
            elif 'renewable_total' in renewable_data.columns:
                renewable_cols.append('renewable_total')
                renewable_data = renewable_data.rename(columns={'renewable_total': 'renewable_generation'})
            
            if len(renewable_cols) > 2:
                merged = pd.merge(merged, renewable_data[renewable_cols], 
                                on=['datetime', 'country'], how='left')
                logger.info(f"Merged renewable data. Shape: {merged.shape}")
        
        # Merge weather data
        if not weather_data.empty:
            merged = pd.merge(merged, weather_data, on=['datetime', 'country'], how='left')
            logger.info(f"Merged weather data. Shape: {merged.shape}")
        
        return merged
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based and engineered features."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Create time features
        df = create_time_features(df, 'datetime')
        logger.info("Created time-based features")
        
        # Create lag features for price (target variable)
        price_lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week
        df = create_lag_features(df, 'price', price_lags)
        logger.info(f"Created {len(price_lags)} price lag features")
        
        # Create rolling features for price
        price_windows = [6, 12, 24, 48, 168]  # 6h to 1 week
        df = create_rolling_features(df, 'price', price_windows)
        logger.info(f"Created rolling features for {len(price_windows)} windows")
        
        # Create load-based features if available
        if 'load' in df.columns:
            load_lags = [1, 24, 168]  # 1h, 1 day, 1 week
            df = create_lag_features(df, 'load', load_lags)
            
            # Load to renewable ratio
            if 'renewable_generation' in df.columns:
                df['renewable_penetration'] = df['renewable_generation'] / (df['load'] + 1e-6)
                df['renewable_penetration'] = df['renewable_penetration'].clip(0, 1)
        
        # Weather-based features
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            # Heating/cooling degree days approximation
            df['heating_degree'] = np.maximum(18 - df['temperature'], 0)
            df['cooling_degree'] = np.maximum(df['temperature'] - 24, 0)
            
            # Wind power proxy (simplified)
            df['wind_power_proxy'] = np.where(
                (df['wind_speed'] >= 3) & (df['wind_speed'] <= 25),
                df['wind_speed'] ** 3,
                0
            )
        
        # Price volatility features
        if 'price' in df.columns:
            df['price_volatility_24h'] = df['price'].rolling(24).std()
            df['price_change_1h'] = df['price'].diff(1)
            df['price_change_24h'] = df['price'].diff(24)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove datetime and country from numeric columns
        numeric_cols = [col for col in numeric_cols if col not in ['datetime', 'country']]
        
        # Handle numeric missing values
        if numeric_cols:
            # Use forward fill for time series data, then backward fill
            df[numeric_cols] = df.groupby('country')[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            
            # For remaining missing values, use median imputation
            remaining_missing = df[numeric_cols].isnull().sum()
            if remaining_missing.sum() > 0:
                logger.info(f"Imputing remaining missing values: {remaining_missing.sum()}")
                imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric'] = imputer
        
        # Handle categorical missing values
        if categorical_cols:
            for col in categorical_cols:
                if col not in ['datetime', 'country']:
                    df[col] = df[col].fillna('unknown')
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_col: str = 'price',
                           scale_features: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for modeling by selecting features and scaling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (prepared DataFrame, feature column names)
        """
        if df.empty:
            return df, []
            
        df = df.copy()
        
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['datetime', 'country', 'currency', 'unit']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure target column is in features
        if target_col not in feature_cols:
            logger.error(f"Target column '{target_col}' not found in data")
            return df, []
        
        # Remove rows with missing target values
        initial_rows = len(df)
        df = df.dropna(subset=[target_col])
        final_rows = len(df)
        
        if initial_rows != final_rows:
            logger.info(f"Removed {initial_rows - final_rows} rows with missing target values")
        
        # Scale features if requested
        if scale_features and len(feature_cols) > 1:
            feature_cols_to_scale = [col for col in feature_cols if col != target_col]
            
            scaler = StandardScaler()
            df[feature_cols_to_scale] = scaler.fit_transform(df[feature_cols_to_scale])
            self.scalers['features'] = scaler
            logger.info(f"Scaled {len(feature_cols_to_scale)} feature columns")
        
        self.feature_columns = feature_cols
        self.target_column = target_col
        
        return df, feature_cols
