"""
Baseline models for Energy Price Forecasting project.
Includes naive forecasting and ARIMA models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import pmdarima as pm  # Temporarily disabled due to numpy compatibility
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from ..utils.helpers import setup_logging

logger = setup_logging()

class NaiveForecaster:
    """
    Naive forecasting methods for baseline comparison.
    """
    
    def __init__(self):
        self.model_type = "naive"
        self.fitted_values = None
        self.seasonal_period = 24  # Daily seasonality for hourly data
    
    def fit(self, y: pd.Series, method: str = "seasonal_naive"):
        """
        Fit naive forecasting model.
        
        Args:
            y: Time series data
            method: Naive method ('naive', 'seasonal_naive', 'drift')
        """
        self.method = method
        self.y_train = y.copy()
        
        if method == "naive":
            # Last value
            self.fitted_values = y.shift(1)
        elif method == "seasonal_naive":
            # Same hour from previous day
            self.fitted_values = y.shift(self.seasonal_period)
        elif method == "drift":
            # Linear trend from first to last value
            n = len(y)
            drift = (y.iloc[-1] - y.iloc[0]) / (n - 1)
            self.fitted_values = y.iloc[0] + drift * np.arange(n)
        
        logger.info(f"Fitted {method} naive forecaster")
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasts
        """
        if self.method == "naive":
            return np.full(steps, self.y_train.iloc[-1])
        
        elif self.method == "seasonal_naive":
            # Repeat last seasonal period
            last_season = self.y_train.iloc[-self.seasonal_period:].values
            n_full_seasons = steps // self.seasonal_period
            remainder = steps % self.seasonal_period
            
            forecasts = np.tile(last_season, n_full_seasons)
            if remainder > 0:
                forecasts = np.concatenate([forecasts, last_season[:remainder]])
            
            return forecasts
        
        elif self.method == "drift":
            # Continue linear trend
            n = len(self.y_train)
            drift = (self.y_train.iloc[-1] - self.y_train.iloc[0]) / (n - 1)
            last_value = self.y_train.iloc[-1]
            
            return last_value + drift * np.arange(1, steps + 1)
    
    def get_fitted_values(self) -> pd.Series:
        """Get fitted values for training period."""
        return self.fitted_values


class ARIMAForecaster:
    """
    ARIMA and Seasonal ARIMA forecasting models.
    """
    
    def __init__(self, seasonal: bool = True, seasonal_period: int = 24):
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.model = None
        self.model_fit = None
        self.order = None
        self.seasonal_order = None
    
    def check_stationarity(self, y: pd.Series) -> Dict[str, any]:
        """
        Check stationarity of time series using Augmented Dickey-Fuller test.
        
        Args:
            y: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        result = adfuller(y.dropna())
        
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        logger.info(f"ADF Test - Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
        logger.info(f"Series is {'stationary' if stationarity_result['is_stationary'] else 'non-stationary'}")
        
        return stationarity_result
    
    def auto_arima_fit(self, y: pd.Series, **kwargs) -> None:
        """
        Automatically determine ARIMA parameters using pmdarima.
        
        Args:
            y: Time series data
            **kwargs: Additional parameters for auto_arima
        """
        logger.info("Starting auto ARIMA parameter selection...")
        
        # Default parameters for auto_arima
        auto_arima_params = {
            'start_p': 0, 'start_q': 0, 'max_p': 5, 'max_q': 5,
            'seasonal': self.seasonal,
            'm': self.seasonal_period if self.seasonal else 1,
            'start_P': 0, 'start_Q': 0, 'max_P': 2, 'max_Q': 2,
            'stepwise': True,
            'suppress_warnings': True,
            'error_action': 'ignore',
            'trace': False
        }
        
        # Update with user-provided parameters
        auto_arima_params.update(kwargs)
        
        # Auto ARIMA temporarily disabled due to dependency issues
        logger.warning("Auto ARIMA temporarily disabled, using fallback ARIMA(1,1,1)")
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, self.seasonal_period) if self.seasonal else None
        logger.info(f"Using fallback ARIMA order: {self.order}")
    
    def fit(self, y: pd.Series, order: Tuple = None, seasonal_order: Tuple = None):
        """
        Fit ARIMA model with specified or auto-determined parameters.
        
        Args:
            y: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        """
        if order is None:
            # Use auto ARIMA
            self.auto_arima_fit(y)
        else:
            self.order = order
            self.seasonal_order = seasonal_order
        
        try:
            # Fit ARIMA model using statsmodels
            if self.seasonal and self.seasonal_order:
                self.model_fit = ARIMA(
                    y.dropna(), 
                    order=self.order, 
                    seasonal_order=self.seasonal_order
                ).fit()
            else:
                self.model_fit = ARIMA(y.dropna(), order=self.order).fit()
            
            logger.info("ARIMA model fitted successfully")
            
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            # Try simpler model
            try:
                self.order = (1, 1, 1)
                self.seasonal_order = None
                self.model_fit = ARIMA(y.dropna(), order=self.order).fit()
                logger.info("Fitted simplified ARIMA(1,1,1) model")
            except Exception as e2:
                logger.error(f"Simplified ARIMA also failed: {e2}")
                raise
    
    def predict(self, steps: int, return_conf_int: bool = False) -> np.ndarray:
        """
        Generate forecasts using fitted ARIMA model.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Forecasts (and confidence intervals if requested)
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            forecast_result = self.model_fit.forecast(steps=steps, alpha=0.05)
            
            if return_conf_int:
                conf_int = self.model_fit.get_forecast(steps=steps).conf_int()
                return forecast_result, conf_int
            else:
                return forecast_result
                
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            # Return naive forecast as fallback
            last_value = self.model_fit.fittedvalues.iloc[-1]
            return np.full(steps, last_value)
    
    def get_fitted_values(self) -> pd.Series:
        """Get fitted values from the model."""
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
        
        return self.model_fit.fittedvalues
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals."""
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
        
        return self.model_fit.resid
    
    def model_diagnostics(self) -> Dict[str, any]:
        """
        Get model diagnostic information.
        
        Returns:
            Dictionary with model diagnostics
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
        
        diagnostics = {
            'aic': self.model_fit.aic,
            'bic': self.model_fit.bic,
            'hqic': self.model_fit.hqic,
            'llf': self.model_fit.llf,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'params': self.model_fit.params.to_dict(),
            'pvalues': self.model_fit.pvalues.to_dict()
        }
        
        return diagnostics


class BaselineEvaluator:
    """
    Evaluator for baseline models.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, y_true: pd.Series, y_pred: pd.Series, 
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Align series and remove NaN values
        aligned_data = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
        
        if len(aligned_data) == 0:
            logger.warning(f"No valid data points for evaluation of {model_name}")
            return {}
        
        y_true_clean = aligned_data['true']
        y_pred_clean = aligned_data['pred']
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        
        # Additional metrics
        bias = np.mean(y_pred_clean - y_true_clean)
        r2 = 1 - (np.sum((y_true_clean - y_pred_clean) ** 2) / 
                  np.sum((y_true_clean - np.mean(y_true_clean)) ** 2))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'bias': bias,
            'r2': r2,
            'n_samples': len(aligned_data)
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(3)
        
        # Rank models by RMSE (lower is better)
        comparison_df['rmse_rank'] = comparison_df['rmse'].rank()
        comparison_df = comparison_df.sort_values('rmse_rank')
        
        return comparison_df
