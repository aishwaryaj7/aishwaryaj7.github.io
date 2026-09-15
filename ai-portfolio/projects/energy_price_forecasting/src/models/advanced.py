"""
Advanced machine learning models for Energy Price Forecasting project.
Includes XGBoost and LightGBM models with time series features.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..utils.helpers import setup_logging

logger = setup_logging()

class TimeSeriesMLModel:
    """
    Base class for time series machine learning models.
    """
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.is_fitted = False
        self.feature_names = []
        
    def create_time_series_features(self, df: pd.DataFrame, target_col: str,
                                  datetime_col: str = 'datetime') -> pd.DataFrame:
        """
        Create time series specific features for ML models.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            datetime_col: Datetime column name
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Ensure datetime column is datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)
        
        # Lag features
        lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        windows = [6, 12, 24, 48, 168]  # 6h to 1 week
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
        
        # Expanding statistics
        df[f'{target_col}_expanding_mean'] = df[target_col].expanding().mean()
        df[f'{target_col}_expanding_std'] = df[target_col].expanding().std()
        
        # Difference features
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_24'] = df[target_col].diff(24)
        df[f'{target_col}_diff_168'] = df[target_col].diff(168)
        
        # Percentage change
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_24'] = df[target_col].pct_change(24)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str,
                        exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (feature DataFrame, feature column names)
        """
        if exclude_cols is None:
            exclude_cols = ['datetime', 'country', 'currency', 'unit']
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [target_col]]
        
        # Remove columns with too many missing values (>50%)
        missing_pct = df[feature_cols].isnull().mean()
        valid_features = missing_pct[missing_pct <= 0.5].index.tolist()
        
        logger.info(f"Selected {len(valid_features)} features out of {len(feature_cols)}")
        
        return df[valid_features + [target_col]], valid_features


class XGBoostForecaster(TimeSeriesMLModel):
    """
    XGBoost model for time series forecasting.
    """
    
    def __init__(self, **xgb_params):
        super().__init__("xgboost")
        
        # Default XGBoost parameters optimized for time series
        self.default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        
        # Update with user parameters
        self.params = {**self.default_params, **xgb_params}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_split: float = 0.2, verbose: bool = True):
        """
        Fit XGBoost model with time series validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress
        """
        # Remove rows with missing target values
        valid_idx = ~y.isnull()
        X_clean = X[valid_idx].copy()
        y_clean = y[valid_idx].copy()
        
        # Handle missing values in features
        X_clean = X_clean.fillna(X_clean.median())
        
        # Time series split for validation
        split_idx = int(len(X_clean) * (1 - validation_split))
        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        
        self.feature_names = X_train.columns.tolist()
        
        # Create XGBoost model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Fit with early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        logger.info("XGBoost model fitted successfully")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using fitted model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle missing values
        X_pred = X[self.feature_names].fillna(X[self.feature_names].median())
        
        return self.model.predict(X_pred)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")
        
        return self.feature_importance.head(top_n)


class LightGBMForecaster(TimeSeriesMLModel):
    """
    LightGBM model for time series forecasting.
    """
    
    def __init__(self, **lgb_params):
        super().__init__("lightgbm")
        
        # Default LightGBM parameters
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        
        self.params = {**self.default_params, **lgb_params}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_split: float = 0.2, verbose: bool = True):
        """
        Fit LightGBM model with time series validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress
        """
        # Remove rows with missing target values
        valid_idx = ~y.isnull()
        X_clean = X[valid_idx].copy()
        y_clean = y[valid_idx].copy()
        
        # Handle missing values in features
        X_clean = X_clean.fillna(X_clean.median())
        
        # Time series split for validation
        split_idx = int(len(X_clean) * (1 - validation_split))
        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        
        self.feature_names = X_train.columns.tolist()
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(self.params['early_stopping_rounds'])] if verbose else None
        )
        
        # Store feature importance
        importance_scores = self.model.feature_importance(importance_type='gain')
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        logger.info("LightGBM model fitted successfully")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using fitted model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle missing values
        X_pred = X[self.feature_names].fillna(X[self.feature_names].median())
        
        return self.model.predict(X_pred, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")
        
        return self.feature_importance.head(top_n)


class ModelEnsemble:
    """
    Ensemble of multiple forecasting models.
    """

    def __init__(self, models: List[TimeSeriesMLModel], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_fitted = False

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        """
        Fit all models in the ensemble.

        Args:
            X: Feature DataFrame
            y: Target series
            **fit_params: Parameters passed to individual model fit methods
        """
        logger.info(f"Fitting ensemble of {len(self.models)} models")

        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {model.model_type}")
            model.fit(X, y, **fit_params)

        self.is_fitted = True
        logger.info("Ensemble fitting completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions as weighted average.

        Args:
            X: Feature DataFrame

        Returns:
            Array of ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_pred

    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual models.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary mapping model types to predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        individual_preds = {}
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            individual_preds[f"{model.model_type}_{i}"] = pred

        return individual_preds
