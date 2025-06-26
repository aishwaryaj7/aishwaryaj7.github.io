"""
Training pipeline for Energy Price Forecasting project.
"""
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.extractors import ENTSOEExtractor, WeatherExtractor
from src.data.transformers import EnergyDataTransformer
from src.data.loaders import GCSDataLoader
from src.models.baseline import NaiveForecaster, ARIMAForecaster, BaselineEvaluator
from src.models.advanced import XGBoostForecaster, LightGBMForecaster, ModelEnsemble
from src.models.evaluation import ForecastEvaluator
from src.analysis.eda import EnergyDataAnalyzer
from src.utils.config import config
from src.utils.helpers import setup_logging

logger = setup_logging()

class EnergyForecastingPipeline:
    """
    Main training pipeline for energy price forecasting.
    """
    
    def __init__(self):
        self.data_loader = GCSDataLoader()
        self.transformer = EnergyDataTransformer()
        self.evaluator = ForecastEvaluator()
        self.analyzer = EnergyDataAnalyzer()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        
        self.models = {}
        self.results = {}
    
    async def extract_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Extract data from all sources.
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            
        Returns:
            Dictionary with extracted data
        """
        logger.info(f"Starting data extraction from {start_date} to {end_date}")
        
        # Initialize extractors
        entso_extractor = ENTSOEExtractor(config.api.entso_e_token)
        weather_extractor = WeatherExtractor(config.api.openweather_token)
        
        extracted_data = {
            'prices': pd.DataFrame(),
            'load': pd.DataFrame(),
            'renewable': pd.DataFrame(),
            'weather': pd.DataFrame()
        }
        
        # Extract data for each country
        for country in config.data.market_zones:
            logger.info(f"Extracting data for {country}")
            
            try:
                # Extract price data
                price_data = await entso_extractor.get_day_ahead_prices(country, start_date, end_date)
                if not price_data.empty:
                    extracted_data['prices'] = pd.concat([extracted_data['prices'], price_data], ignore_index=True)
                    
                    # Save raw price data
                    self.data_loader.save_raw_data(price_data, 'prices', country, start_date)
                
                # Extract load data
                load_data = await entso_extractor.get_actual_load(country, start_date, end_date)
                if not load_data.empty:
                    extracted_data['load'] = pd.concat([extracted_data['load'], load_data], ignore_index=True)
                    
                    # Save raw load data
                    self.data_loader.save_raw_data(load_data, 'load', country, start_date)
                
                # Extract renewable data
                renewable_data = await entso_extractor.get_renewable_generation(country, start_date, end_date)
                if not renewable_data.empty:
                    extracted_data['renewable'] = pd.concat([extracted_data['renewable'], renewable_data], ignore_index=True)
                    
                    # Save raw renewable data
                    self.data_loader.save_raw_data(renewable_data, 'renewable', country, start_date)
                
                # Extract weather data
                weather_data = await weather_extractor.get_historical_weather(country, start_date, end_date)
                if not weather_data.empty:
                    extracted_data['weather'] = pd.concat([extracted_data['weather'], weather_data], ignore_index=True)
                    
                    # Save raw weather data
                    self.data_loader.save_raw_data(weather_data, 'weather', country, start_date)
                
            except Exception as e:
                logger.error(f"Error extracting data for {country}: {e}")
                continue
        
        logger.info("Data extraction completed")
        return extracted_data
    
    def transform_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform and combine raw data.
        
        Args:
            raw_data: Dictionary with raw data
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformation")
        
        # Transform data
        transformed_data = self.transformer.transform_raw_data(
            raw_data['prices'],
            raw_data['load'],
            raw_data['renewable'],
            raw_data['weather']
        )
        
        # Save processed data
        if not transformed_data.empty:
            self.data_loader.save_processed_data(transformed_data, 'cleaned')
        
        # Prepare for modeling
        final_data, feature_cols = self.transformer.prepare_for_modeling(transformed_data)
        
        if not final_data.empty:
            self.data_loader.save_processed_data(final_data, 'final')
        
        logger.info(f"Data transformation completed. Final shape: {final_data.shape}")
        return final_data
    
    def perform_eda(self, data: pd.DataFrame) -> Dict:
        """
        Perform exploratory data analysis.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting exploratory data analysis")
        
        self.analyzer.load_data(data)
        
        # Run all analysis methods
        basic_stats = self.analyzer.basic_statistics()
        price_analysis = self.analyzer.price_analysis()
        seasonal_decomp = self.analyzer.seasonal_decomposition()
        stationarity = self.analyzer.stationarity_analysis()
        correlations = self.analyzer.correlation_analysis()
        autocorr = self.analyzer.autocorrelation_analysis()
        
        # Generate summary report
        summary_report = self.analyzer.generate_summary_report()
        
        logger.info("EDA completed")
        logger.info("\n" + summary_report)
        
        return {
            'basic_stats': basic_stats,
            'price_analysis': price_analysis,
            'seasonal_decomposition': seasonal_decomp,
            'stationarity': stationarity,
            'correlations': correlations,
            'autocorrelation': autocorr,
            'summary_report': summary_report
        }
    
    def train_baseline_models(self, data: pd.DataFrame) -> Dict:
        """
        Train baseline forecasting models.
        
        Args:
            data: Training data
            
        Returns:
            Dictionary with trained baseline models
        """
        logger.info("Training baseline models")
        
        # Prepare time series data
        ts_data = data.set_index('datetime')['price'].dropna()
        
        baseline_models = {}
        baseline_evaluator = BaselineEvaluator()
        
        # Train Naive models
        for method in ['naive', 'seasonal_naive', 'drift']:
            try:
                model = NaiveForecaster()
                model.fit(ts_data, method=method)
                baseline_models[f'naive_{method}'] = model
                
                # Evaluate on training data
                fitted_values = model.get_fitted_values()
                if fitted_values is not None and not fitted_values.empty:
                    baseline_evaluator.evaluate_model(ts_data, fitted_values, f'naive_{method}')
                
            except Exception as e:
                logger.error(f"Error training {method} naive model: {e}")
        
        # Train ARIMA model
        try:
            arima_model = ARIMAForecaster(seasonal=True)
            arima_model.fit(ts_data)
            baseline_models['arima'] = arima_model
            
            # Evaluate ARIMA
            fitted_values = arima_model.get_fitted_values()
            if fitted_values is not None and not fitted_values.empty:
                baseline_evaluator.evaluate_model(ts_data, fitted_values, 'arima')
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
        
        # Compare baseline models
        comparison = baseline_evaluator.compare_models()
        logger.info("Baseline model comparison:")
        logger.info(f"\n{comparison}")
        
        self.models.update(baseline_models)
        self.results['baseline_comparison'] = comparison
        
        return baseline_models
    
    def train_advanced_models(self, data: pd.DataFrame) -> Dict:
        """
        Train advanced ML models.
        
        Args:
            data: Training data
            
        Returns:
            Dictionary with trained advanced models
        """
        logger.info("Training advanced ML models")
        
        # Prepare features
        feature_cols = [col for col in data.columns 
                       if col not in ['datetime', 'country', 'currency', 'unit']]
        
        X = data[feature_cols].drop(columns=['price'])
        y = data['price']
        
        # Remove rows with missing target
        valid_idx = ~y.isnull()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        advanced_models = {}
        
        # Train XGBoost
        try:
            with mlflow.start_run(run_name="xgboost_training"):
                xgb_model = XGBoostForecaster()
                xgb_model.fit(X_clean, y_clean)
                advanced_models['xgboost'] = xgb_model
                
                # Log model and metrics
                mlflow.log_params(xgb_model.params)
                
                # Generate predictions for evaluation
                predictions = xgb_model.predict(X_clean)
                metrics = self.evaluator.calculate_metrics(y_clean, pd.Series(predictions), 'xgboost')
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log feature importance
                feature_importance = xgb_model.get_feature_importance()
                mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
                
                # Log model
                mlflow.xgboost.log_model(xgb_model.model, "model")
                
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
        
        # Train LightGBM
        try:
            with mlflow.start_run(run_name="lightgbm_training"):
                lgb_model = LightGBMForecaster()
                lgb_model.fit(X_clean, y_clean)
                advanced_models['lightgbm'] = lgb_model
                
                # Log model and metrics
                mlflow.log_params(lgb_model.params)
                
                # Generate predictions for evaluation
                predictions = lgb_model.predict(X_clean)
                metrics = self.evaluator.calculate_metrics(y_clean, pd.Series(predictions), 'lightgbm')
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log feature importance
                feature_importance = lgb_model.get_feature_importance()
                mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
                
                # Log model
                mlflow.lightgbm.log_model(lgb_model.model, "model")
                
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
        
        # Create ensemble if multiple models trained
        if len(advanced_models) > 1:
            try:
                ensemble = ModelEnsemble(list(advanced_models.values()))
                ensemble.fit(X_clean, y_clean, verbose=False)
                advanced_models['ensemble'] = ensemble
                
                # Evaluate ensemble
                predictions = ensemble.predict(X_clean)
                self.evaluator.calculate_metrics(y_clean, pd.Series(predictions), 'ensemble')
                
            except Exception as e:
                logger.error(f"Error creating ensemble model: {e}")
        
        self.models.update(advanced_models)
        
        return advanced_models
    
    def save_models(self):
        """Save all trained models."""
        logger.info("Saving trained models")
        
        # Prepare model artifacts
        artifacts = {
            'models': self.models,
            'transformer': self.transformer,
            'feature_columns': self.transformer.feature_columns,
            'target_column': self.transformer.target_column,
            'training_timestamp': datetime.now(),
            'config': config
        }
        
        # Save to cloud storage
        model_path = self.data_loader.save_model_artifacts(artifacts, 'energy_forecasting')
        logger.info(f"Models saved to: {model_path}")
        
        return model_path
    
    async def run_full_pipeline(self, start_date: datetime = None, end_date: datetime = None):
        """
        Run the complete training pipeline.
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
        """
        logger.info("Starting full training pipeline")
        
        # Default date range (last 30 days)
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        try:
            # Step 1: Extract data
            raw_data = await self.extract_data(start_date, end_date)
            
            # Step 2: Transform data
            processed_data = self.transform_data(raw_data)
            
            if processed_data.empty:
                logger.error("No processed data available. Pipeline stopped.")
                return
            
            # Step 3: Perform EDA
            eda_results = self.perform_eda(processed_data)
            
            # Step 4: Train baseline models
            baseline_models = self.train_baseline_models(processed_data)
            
            # Step 5: Train advanced models
            advanced_models = self.train_advanced_models(processed_data)
            
            # Step 6: Save models
            model_path = self.save_models()
            
            logger.info("Full training pipeline completed successfully")
            logger.info(f"Models saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    # Run the training pipeline
    pipeline = EnergyForecastingPipeline()
    
    # Run with default date range
    asyncio.run(pipeline.run_full_pipeline())
