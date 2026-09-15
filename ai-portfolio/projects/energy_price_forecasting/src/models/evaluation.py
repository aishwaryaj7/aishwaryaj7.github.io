"""
Model evaluation module for Energy Price Forecasting project.
Includes comprehensive evaluation metrics and backtesting functionality.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..utils.helpers import setup_logging

logger = setup_logging()

class ForecastEvaluator:
    """
    Comprehensive evaluation for time series forecasting models.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.backtest_results = {}
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, 
                         model_name: str = "model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
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
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Additional metrics
        bias = np.mean(y_pred_clean - y_true_clean)
        max_error = np.max(np.abs(y_true_clean - y_pred_clean))
        
        # Directional accuracy (for price movements)
        if len(y_true_clean) > 1:
            true_direction = np.sign(y_true_clean.diff().dropna())
            pred_direction = np.sign(y_pred_clean.diff().dropna())
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = np.nan
        
        # Pinball loss for quantile forecasts (assuming median forecast)
        quantile = 0.5
        pinball_loss = np.mean(np.maximum(
            quantile * (y_true_clean - y_pred_clean),
            (quantile - 1) * (y_true_clean - y_pred_clean)
        ))
        
        # Normalized metrics
        naive_forecast = y_true_clean.shift(1).dropna()
        if len(naive_forecast) > 0:
            naive_mae = mean_absolute_error(y_true_clean[1:], naive_forecast)
            normalized_mae = mae / naive_mae if naive_mae > 0 else np.inf
        else:
            normalized_mae = np.nan
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'bias': bias,
            'max_error': max_error,
            'directional_accuracy': directional_accuracy,
            'pinball_loss': pinball_loss,
            'normalized_mae': normalized_mae,
            'n_samples': len(aligned_data)
        }
        
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.3f}")
        
        return metrics
    
    def rolling_origin_backtest(self, data: pd.DataFrame, model, 
                              target_col: str = 'price',
                              datetime_col: str = 'datetime',
                              initial_window: int = 168,  # 1 week
                              forecast_horizon: int = 24,  # 1 day
                              step_size: int = 24) -> Dict:
        """
        Perform rolling origin backtesting.
        
        Args:
            data: Time series data
            model: Fitted model object
            target_col: Target column name
            datetime_col: Datetime column name
            initial_window: Initial training window size
            forecast_horizon: Number of steps to forecast
            step_size: Step size for rolling window
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting rolling origin backtesting...")
        
        data = data.sort_values(datetime_col).reset_index(drop=True)
        
        backtest_results = {
            'forecasts': [],
            'actuals': [],
            'dates': [],
            'metrics_by_window': []
        }
        
        # Prepare feature columns
        feature_cols = [col for col in data.columns 
                       if col not in [datetime_col, 'country', 'currency', 'unit']]
        
        start_idx = initial_window
        end_idx = len(data) - forecast_horizon
        
        window_count = 0
        
        for i in range(start_idx, end_idx, step_size):
            try:
                # Training data
                train_data = data.iloc[i-initial_window:i]
                X_train = train_data[feature_cols].drop(columns=[target_col])
                y_train = train_data[target_col]
                
                # Test data
                test_data = data.iloc[i:i+forecast_horizon]
                X_test = test_data[feature_cols].drop(columns=[target_col])
                y_test = test_data[target_col]
                
                # Fit model on training data
                model.fit(X_train, y_train, verbose=False)
                
                # Generate forecasts
                forecasts = model.predict(X_test)
                
                # Store results
                backtest_results['forecasts'].extend(forecasts)
                backtest_results['actuals'].extend(y_test.values)
                backtest_results['dates'].extend(test_data[datetime_col].values)
                
                # Calculate metrics for this window
                window_metrics = self.calculate_metrics(
                    pd.Series(y_test.values), 
                    pd.Series(forecasts),
                    f"window_{window_count}"
                )
                window_metrics['window_start'] = train_data[datetime_col].iloc[0]
                window_metrics['window_end'] = train_data[datetime_col].iloc[-1]
                window_metrics['forecast_start'] = test_data[datetime_col].iloc[0]
                window_metrics['forecast_end'] = test_data[datetime_col].iloc[-1]
                
                backtest_results['metrics_by_window'].append(window_metrics)
                
                window_count += 1
                
                if window_count % 10 == 0:
                    logger.info(f"Completed {window_count} backtest windows")
                
            except Exception as e:
                logger.warning(f"Error in backtest window {window_count}: {e}")
                continue
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(
            pd.Series(backtest_results['actuals']),
            pd.Series(backtest_results['forecasts']),
            "backtest_overall"
        )
        
        backtest_results['overall_metrics'] = overall_metrics
        backtest_results['n_windows'] = window_count
        
        logger.info(f"Backtesting completed with {window_count} windows")
        logger.info(f"Overall RMSE: {overall_metrics.get('rmse', 'N/A'):.2f}")
        
        return backtest_results
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models' performance.
        
        Args:
            results_dict: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with model comparison
        """
        if not results_dict:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(results_dict).T
        
        # Round numeric columns
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        comparison_df[numeric_cols] = comparison_df[numeric_cols].round(3)
        
        # Rank models by RMSE (lower is better)
        if 'rmse' in comparison_df.columns:
            comparison_df['rmse_rank'] = comparison_df['rmse'].rank()
            comparison_df = comparison_df.sort_values('rmse_rank')
        
        return comparison_df
    
    def plot_forecast_vs_actual(self, y_true: pd.Series, y_pred: pd.Series,
                               dates: pd.Series = None, model_name: str = "Model",
                               save_path: str = None) -> plt.Figure:
        """
        Plot forecast vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Date series for x-axis
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Align data
        aligned_data = pd.DataFrame({
            'true': y_true, 
            'pred': y_pred,
            'dates': dates if dates is not None else range(len(y_true))
        }).dropna()
        
        # Time series plot
        ax1.plot(aligned_data['dates'], aligned_data['true'], 
                label='Actual', alpha=0.7, linewidth=1)
        ax1.plot(aligned_data['dates'], aligned_data['pred'], 
                label='Forecast', alpha=0.7, linewidth=1)
        ax1.set_title(f'{model_name} - Forecast vs Actual')
        ax1.set_ylabel('Price (EUR/MWh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(aligned_data['true'], aligned_data['pred'], alpha=0.5)
        
        # Perfect prediction line
        min_val = min(aligned_data['true'].min(), aligned_data['pred'].min())
        max_val = max(aligned_data['true'].max(), aligned_data['pred'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax2.set_xlabel('Actual Price (EUR/MWh)')
        ax2.set_ylabel('Predicted Price (EUR/MWh)')
        ax2.set_title('Predicted vs Actual Scatter Plot')
        ax2.grid(True, alpha=0.3)
        
        # Add R² to scatter plot
        if len(aligned_data) > 1:
            r2 = r2_score(aligned_data['true'], aligned_data['pred'])
            ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_residuals(self, y_true: pd.Series, y_pred: pd.Series,
                      dates: pd.Series = None, model_name: str = "Model",
                      save_path: str = None) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Date series for x-axis
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Calculate residuals
        aligned_data = pd.DataFrame({
            'true': y_true, 
            'pred': y_pred,
            'dates': dates if dates is not None else range(len(y_true))
        }).dropna()
        
        residuals = aligned_data['true'] - aligned_data['pred']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        ax1.plot(aligned_data['dates'], residuals, alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax1.set_title('Residuals Over Time')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Residuals vs fitted values
        ax3.scatter(aligned_data['pred'], residuals, alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals vs Fitted Values')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot (simplified)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        return fig
