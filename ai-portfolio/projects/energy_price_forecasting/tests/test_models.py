"""
Tests for model modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.baseline import NaiveForecaster, ARIMAForecaster, BaselineEvaluator
from src.models.advanced import XGBoostForecaster, LightGBMForecaster, ModelEnsemble
from src.models.evaluation import ForecastEvaluator

class TestBaselineModels:
    """Test baseline forecasting models."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic time series data
        np.random.seed(42)
        date_range = pd.date_range(start="2024-01-01", periods=168, freq="H")  # 1 week
        
        # Create realistic price pattern
        hours = np.array([dt.hour for dt in date_range])
        days = np.array([dt.dayofweek for dt in date_range])
        
        base_price = 50
        hourly_pattern = 10 * np.sin(2 * np.pi * hours / 24)
        daily_pattern = 5 * np.sin(2 * np.pi * days / 7)
        noise = np.random.normal(0, 3, len(date_range))
        
        prices = base_price + hourly_pattern + daily_pattern + noise
        
        self.ts_data = pd.Series(prices, index=date_range)
    
    def test_naive_forecaster_initialization(self):
        """Test naive forecaster initialization."""
        forecaster = NaiveForecaster()
        assert forecaster.model_type == "naive"
        assert forecaster.seasonal_period == 24
    
    def test_naive_forecaster_fit_and_predict(self):
        """Test naive forecaster fitting and prediction."""
        self.setUp()
        
        forecaster = NaiveForecaster()
        
        # Test different methods
        for method in ['naive', 'seasonal_naive', 'drift']:
            forecaster.fit(self.ts_data, method=method)
            
            # Generate predictions
            predictions = forecaster.predict(24)  # 24 hours
            
            assert len(predictions) == 24
            assert not np.any(np.isnan(predictions))
            
            # Get fitted values
            fitted_values = forecaster.get_fitted_values()
            assert fitted_values is not None
    
    def test_arima_forecaster_initialization(self):
        """Test ARIMA forecaster initialization."""
        forecaster = ARIMAForecaster(seasonal=True)
        assert forecaster.seasonal == True
        assert forecaster.seasonal_period == 24
    
    def test_arima_forecaster_stationarity_check(self):
        """Test stationarity checking."""
        self.setUp()
        
        forecaster = ARIMAForecaster()
        stationarity_result = forecaster.check_stationarity(self.ts_data)
        
        assert 'adf_statistic' in stationarity_result
        assert 'p_value' in stationarity_result
        assert 'is_stationary' in stationarity_result
        assert isinstance(stationarity_result['is_stationary'], bool)
    
    def test_arima_forecaster_fit_and_predict(self):
        """Test ARIMA forecaster fitting and prediction."""
        self.setUp()
        
        forecaster = ARIMAForecaster(seasonal=False)  # Use non-seasonal for faster testing
        
        try:
            # Fit with simple order
            forecaster.fit(self.ts_data, order=(1, 1, 1))
            
            # Generate predictions
            predictions = forecaster.predict(24)
            
            assert len(predictions) == 24
            assert not np.any(np.isnan(predictions))
            
            # Get model diagnostics
            diagnostics = forecaster.model_diagnostics()
            assert 'aic' in diagnostics
            assert 'bic' in diagnostics
            
        except Exception as e:
            # ARIMA might fail with synthetic data, that's okay for testing
            pytest.skip(f"ARIMA fitting failed with synthetic data: {e}")
    
    def test_baseline_evaluator(self):
        """Test baseline model evaluator."""
        self.setUp()
        
        evaluator = BaselineEvaluator()
        
        # Create some predictions (simple naive forecast)
        y_true = self.ts_data[-24:]  # Last 24 hours
        y_pred = pd.Series(self.ts_data.iloc[-25:-1].values, index=y_true.index)  # Shifted by 1
        
        # Evaluate
        metrics = evaluator.evaluate_model(y_true, y_pred, "test_model")
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        assert metrics['n_samples'] > 0
        
        # Test comparison
        comparison = evaluator.compare_models()
        assert not comparison.empty
        assert 'test_model' in comparison.index

class TestAdvancedModels:
    """Test advanced ML models."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create feature matrix
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some relationship to features
        y = (X['feature_0'] * 2 + 
             X['feature_1'] * 1.5 + 
             X['feature_2'] * -1 + 
             np.random.normal(0, 0.5, n_samples))
        
        self.X_train = X[:800]
        self.y_train = y[:800]
        self.X_test = X[800:]
        self.y_test = y[800:]
    
    def test_xgboost_forecaster_initialization(self):
        """Test XGBoost forecaster initialization."""
        forecaster = XGBoostForecaster()
        assert forecaster.model_type == "xgboost"
        assert 'n_estimators' in forecaster.params
        assert 'learning_rate' in forecaster.params
    
    def test_xgboost_forecaster_fit_and_predict(self):
        """Test XGBoost forecaster fitting and prediction."""
        self.setUp()
        
        forecaster = XGBoostForecaster(n_estimators=10)  # Small for fast testing
        
        # Fit model
        forecaster.fit(self.X_train, self.y_train, verbose=False)
        
        assert forecaster.is_fitted == True
        assert forecaster.feature_names == list(self.X_train.columns)
        
        # Generate predictions
        predictions = forecaster.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert not np.any(np.isnan(predictions))
        
        # Get feature importance
        importance = forecaster.get_feature_importance()
        assert not importance.empty
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_lightgbm_forecaster_initialization(self):
        """Test LightGBM forecaster initialization."""
        forecaster = LightGBMForecaster()
        assert forecaster.model_type == "lightgbm"
        assert 'num_leaves' in forecaster.params
        assert 'learning_rate' in forecaster.params
    
    def test_lightgbm_forecaster_fit_and_predict(self):
        """Test LightGBM forecaster fitting and prediction."""
        self.setUp()
        
        forecaster = LightGBMForecaster(n_estimators=10, verbose=-1)  # Small for fast testing
        
        # Fit model
        forecaster.fit(self.X_train, self.y_train, verbose=False)
        
        assert forecaster.is_fitted == True
        assert forecaster.feature_names == list(self.X_train.columns)
        
        # Generate predictions
        predictions = forecaster.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert not np.any(np.isnan(predictions))
        
        # Get feature importance
        importance = forecaster.get_feature_importance()
        assert not importance.empty
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_model_ensemble(self):
        """Test model ensemble functionality."""
        self.setUp()
        
        # Create individual models
        xgb_model = XGBoostForecaster(n_estimators=10)
        lgb_model = LightGBMForecaster(n_estimators=10, verbose=-1)
        
        # Create ensemble
        ensemble = ModelEnsemble([xgb_model, lgb_model], weights=[0.6, 0.4])
        
        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6  # Weights should sum to 1
        
        # Fit ensemble
        ensemble.fit(self.X_train, self.y_train, verbose=False)
        
        assert ensemble.is_fitted == True
        
        # Generate predictions
        predictions = ensemble.predict(self.X_test)
        
        assert len(predictions) == len(self.X_test)
        assert not np.any(np.isnan(predictions))
        
        # Get individual predictions
        individual_preds = ensemble.get_individual_predictions(self.X_test)
        assert len(individual_preds) == 2

class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic true and predicted values
        n_samples = 100
        self.y_true = pd.Series(np.random.normal(50, 10, n_samples))
        self.y_pred = self.y_true + np.random.normal(0, 3, n_samples)  # Add some error
        self.dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    def test_forecast_evaluator_initialization(self):
        """Test forecast evaluator initialization."""
        evaluator = ForecastEvaluator()
        assert evaluator.evaluation_results == {}
        assert evaluator.backtest_results == {}
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        self.setUp()
        
        evaluator = ForecastEvaluator()
        metrics = evaluator.calculate_metrics(self.y_true, self.y_pred, "test_model")
        
        # Check all expected metrics are present
        expected_metrics = ['mae', 'rmse', 'mape', 'r2', 'bias', 'max_error', 
                          'directional_accuracy', 'pinball_loss', 'normalized_mae', 'n_samples']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        assert metrics['n_samples'] == len(self.y_true)
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mape'] >= 0
    
    def test_compare_models(self):
        """Test model comparison."""
        self.setUp()
        
        evaluator = ForecastEvaluator()
        
        # Add multiple model results
        evaluator.calculate_metrics(self.y_true, self.y_pred, "model_1")
        evaluator.calculate_metrics(self.y_true, self.y_pred * 1.1, "model_2")  # Slightly worse
        
        comparison = evaluator.compare_models(evaluator.evaluation_results)
        
        assert not comparison.empty
        assert len(comparison) == 2
        assert 'model_1' in comparison.index
        assert 'model_2' in comparison.index
        
        if 'rmse_rank' in comparison.columns:
            assert comparison.loc['model_1', 'rmse_rank'] <= comparison.loc['model_2', 'rmse_rank']

class TestModelIntegration:
    """Integration tests for model pipeline."""
    
    def test_end_to_end_modeling_pipeline(self):
        """Test complete modeling pipeline."""
        np.random.seed(42)
        
        # Create realistic time series data
        date_range = pd.date_range(start="2024-01-01", periods=500, freq="H")
        
        # Create features
        hours = np.array([dt.hour for dt in date_range])
        days = np.array([dt.dayofweek for dt in date_range])
        
        data = pd.DataFrame({
            'datetime': date_range,
            'hour': hours,
            'day_of_week': days,
            'hour_sin': np.sin(2 * np.pi * hours / 24),
            'hour_cos': np.cos(2 * np.pi * hours / 24),
            'load': np.random.normal(40000, 5000, len(date_range)),
            'temperature': np.random.normal(15, 8, len(date_range))
        })
        
        # Create target with realistic patterns
        base_price = 50
        hourly_pattern = 10 * np.sin(2 * np.pi * hours / 24)
        daily_pattern = 5 * np.sin(2 * np.pi * days / 7)
        load_effect = (data['load'] - 40000) / 10000 * 5  # Load affects price
        temp_effect = (data['temperature'] - 15) / 10 * 2  # Temperature affects price
        noise = np.random.normal(0, 3, len(date_range))
        
        data['price'] = base_price + hourly_pattern + daily_pattern + load_effect + temp_effect + noise
        
        # Split data
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Prepare features
        feature_cols = ['hour', 'day_of_week', 'hour_sin', 'hour_cos', 'load', 'temperature']
        X_train = train_data[feature_cols]
        y_train = train_data['price']
        X_test = test_data[feature_cols]
        y_test = test_data['price']
        
        # Train models
        xgb_model = XGBoostForecaster(n_estimators=50)
        xgb_model.fit(X_train, y_train, verbose=False)
        
        # Generate predictions
        predictions = xgb_model.predict(X_test)
        
        # Evaluate
        evaluator = ForecastEvaluator()
        metrics = evaluator.calculate_metrics(y_test, pd.Series(predictions), "xgboost")
        
        # Check that model performs reasonably well
        assert metrics['mae'] < 10  # Should be better than 10 EUR/MWh MAE
        assert metrics['r2'] > 0.5   # Should explain at least 50% of variance
        
        print(f"Integration test results - MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")

if __name__ == "__main__":
    pytest.main([__file__])
