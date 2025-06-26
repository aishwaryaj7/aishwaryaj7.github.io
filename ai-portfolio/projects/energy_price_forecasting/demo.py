"""
Demo script for Energy Price Forecasting project.
Demonstrates key functionality without requiring external APIs or cloud setup.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.extractors import WeatherExtractor
from src.data.transformers import EnergyDataTransformer
from src.data.loaders import GCSDataLoader
from src.models.baseline import NaiveForecaster, BaselineEvaluator
from src.models.advanced import XGBoostForecaster
from src.models.evaluation import ForecastEvaluator
from src.analysis.eda import EnergyDataAnalyzer
from src.utils.helpers import setup_logging

logger = setup_logging()

def create_synthetic_data(days=30):
    """Create synthetic energy market data for demonstration."""
    logger.info(f"Creating {days} days of synthetic data...")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate realistic price patterns
    hours = np.array([dt.hour for dt in date_range])
    days_of_week = np.array([dt.dayofweek for dt in date_range])
    day_of_year = np.array([dt.timetuple().tm_yday for dt in date_range])
    
    # Base price with multiple patterns
    base_price = 50
    
    # Daily pattern (peak hours)
    daily_pattern = 15 * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Weekly pattern (lower weekend prices)
    weekly_pattern = -8 * np.where(days_of_week >= 5, 1, 0)
    
    # Seasonal pattern
    seasonal_pattern = 10 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 2, len(date_range)))
    random_walk = random_walk - np.mean(random_walk)  # Center around 0
    
    # Combine patterns with noise
    prices = (base_price + daily_pattern + weekly_pattern + 
             seasonal_pattern + random_walk + 
             np.random.normal(0, 5, len(date_range)))
    
    # Create load data (correlated with prices)
    base_load = 40000
    load_daily_pattern = 8000 * np.sin(2 * np.pi * (hours - 8) / 24)
    load_weekly_pattern = -5000 * np.where(days_of_week >= 5, 1, 0)
    load = (base_load + load_daily_pattern + load_weekly_pattern + 
           np.random.normal(0, 2000, len(date_range)))
    
    # Create renewable generation (inversely correlated with prices)
    base_renewable = 15000
    renewable_daily = 5000 * np.sin(2 * np.pi * (hours - 12) / 24)  # Peak at noon
    renewable = np.maximum(0, base_renewable + renewable_daily + 
                          np.random.normal(0, 3000, len(date_range)))
    
    # Create weather data
    base_temp = 15
    temp_seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365)
    temp_daily = 5 * np.sin(2 * np.pi * (hours - 14) / 24)  # Peak at 2pm
    temperature = (base_temp + temp_seasonal + temp_daily + 
                  np.random.normal(0, 3, len(date_range)))
    
    wind_speed = np.maximum(0, 8 + 4 * np.sin(2 * np.pi * day_of_year / 365) + 
                           np.random.normal(0, 3, len(date_range)))
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': date_range,
        'price': prices,
        'country': 'DE',
        'currency': 'EUR/MWh',
        'load': load,
        'renewable_generation': renewable,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'cloud_cover': np.random.randint(0, 101, len(date_range)),
        'humidity': np.random.randint(30, 90, len(date_range))
    })
    
    logger.info(f"✅ Created synthetic data with {len(data)} records")
    return data

async def demo_data_extraction():
    """Demonstrate data extraction capabilities."""
    logger.info("🔄 Demonstrating data extraction...")
    
    # Demo weather extractor (creates synthetic data)
    weather_extractor = WeatherExtractor("demo_key")
    
    start_date = datetime.now() - timedelta(days=2)
    end_date = datetime.now()
    
    weather_data = await weather_extractor.get_historical_weather("DE", start_date, end_date)
    
    logger.info(f"✅ Extracted {len(weather_data)} weather records")
    return weather_data

def demo_data_transformation(raw_data):
    """Demonstrate data transformation pipeline."""
    logger.info("🔄 Demonstrating data transformation...")
    
    transformer = EnergyDataTransformer()
    
    # Split data into components (simulating different sources)
    price_data = raw_data[['datetime', 'price', 'country', 'currency']].copy()
    load_data = raw_data[['datetime', 'load', 'country']].copy()
    load_data['unit'] = 'MW'
    renewable_data = raw_data[['datetime', 'renewable_generation', 'country']].copy()
    renewable_data['unit'] = 'MW'
    weather_data = raw_data[['datetime', 'temperature', 'wind_speed', 'cloud_cover', 'humidity', 'country']].copy()
    
    # Transform data
    transformed_data = transformer.transform_raw_data(
        price_data, load_data, renewable_data, weather_data
    )
    
    # Prepare for modeling
    final_data, feature_cols = transformer.prepare_for_modeling(transformed_data)
    
    logger.info(f"✅ Transformed data: {final_data.shape}, Features: {len(feature_cols)}")
    return final_data, feature_cols

def demo_eda(data):
    """Demonstrate exploratory data analysis."""
    logger.info("🔄 Demonstrating EDA...")
    
    analyzer = EnergyDataAnalyzer()
    analyzer.load_data(data)
    
    # Run analysis
    basic_stats = analyzer.basic_statistics()
    price_analysis = analyzer.price_analysis()
    stationarity = analyzer.stationarity_analysis()
    correlations = analyzer.correlation_analysis()
    
    # Generate summary
    summary = analyzer.generate_summary_report()
    
    logger.info("✅ EDA completed")
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*60)
    print(summary)
    
    return {
        'basic_stats': basic_stats,
        'price_analysis': price_analysis,
        'stationarity': stationarity,
        'correlations': correlations
    }

def demo_baseline_models(data):
    """Demonstrate baseline model training."""
    logger.info("🔄 Demonstrating baseline models...")
    
    # Prepare time series
    ts_data = data.set_index('datetime')['price'].dropna()
    
    # Train naive models
    evaluator = BaselineEvaluator()
    models = {}
    
    for method in ['naive', 'seasonal_naive', 'drift']:
        try:
            model = NaiveForecaster()
            model.fit(ts_data, method=method)
            models[method] = model
            
            # Evaluate
            fitted_values = model.get_fitted_values()
            if fitted_values is not None and not fitted_values.empty:
                evaluator.evaluate_model(ts_data, fitted_values, method)
                
        except Exception as e:
            logger.warning(f"Failed to train {method} model: {e}")
    
    # Compare models
    comparison = evaluator.compare_models()
    
    logger.info("✅ Baseline models trained")
    if not comparison.empty:
        print("\n" + "="*60)
        print("BASELINE MODEL COMPARISON")
        print("="*60)
        print(comparison.to_string())
    
    return models, comparison

def demo_advanced_models(data, feature_cols):
    """Demonstrate advanced model training."""
    logger.info("🔄 Demonstrating advanced models...")
    
    # Prepare features
    X = data[feature_cols].drop(columns=['price'])
    y = data['price']
    
    # Remove missing values
    valid_idx = ~y.isnull()
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    if len(X_clean) < 100:
        logger.warning("Insufficient data for advanced models")
        return None, None
    
    # Train XGBoost (small model for demo)
    try:
        xgb_model = XGBoostForecaster(n_estimators=50, max_depth=4)
        xgb_model.fit(X_clean, y_clean, verbose=False)
        
        # Generate predictions
        predictions = xgb_model.predict(X_clean)
        
        # Evaluate
        evaluator = ForecastEvaluator()
        metrics = evaluator.calculate_metrics(y_clean, pd.Series(predictions), 'xgboost')
        
        # Feature importance
        importance = xgb_model.get_feature_importance(top_n=10)
        
        logger.info("✅ Advanced model trained")
        print("\n" + "="*60)
        print("XGBOOST MODEL PERFORMANCE")
        print("="*60)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                print(f"{metric.upper()}: {value:.3f}")
        
        print("\n" + "="*40)
        print("TOP 10 FEATURE IMPORTANCE")
        print("="*40)
        print(importance.to_string(index=False))
        
        return xgb_model, metrics
        
    except Exception as e:
        logger.error(f"Failed to train XGBoost model: {e}")
        return None, None

def demo_data_storage():
    """Demonstrate data storage capabilities."""
    logger.info("🔄 Demonstrating data storage...")
    
    loader = GCSDataLoader()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=24, freq='H'),
        'price': np.random.normal(50, 10, 24),
        'country': 'DE'
    })
    
    # Save data (will use local fallback)
    saved_path = loader.save_raw_data(sample_data, 'demo', 'DE', datetime.now())
    
    logger.info(f"✅ Data saved to: {saved_path}")
    return saved_path

async def main():
    """Run the complete demo."""
    print("🚀 Energy Price Forecasting - Project Demo")
    print("="*60)
    
    try:
        # 1. Create synthetic data
        raw_data = create_synthetic_data(days=30)
        
        # 2. Demo data extraction
        weather_data = await demo_data_extraction()
        
        # 3. Demo data transformation
        processed_data, feature_cols = demo_data_transformation(raw_data)
        
        # 4. Demo EDA
        eda_results = demo_eda(processed_data)
        
        # 5. Demo baseline models
        baseline_models, baseline_comparison = demo_baseline_models(processed_data)
        
        # 6. Demo advanced models
        advanced_model, advanced_metrics = demo_advanced_models(processed_data, feature_cols)
        
        # 7. Demo data storage
        storage_path = demo_data_storage()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ Data extraction and transformation")
        print("✅ Exploratory data analysis")
        print("✅ Baseline model training")
        print("✅ Advanced model training")
        print("✅ Data storage and loading")
        print("\n📋 Next steps:")
        print("1. Set up GCP resources: ./scripts/setup_gcp.sh")
        print("2. Configure API keys in environment")
        print("3. Run full training pipeline: python train.py")
        print("4. Start API service: uvicorn api.main:app --reload")
        print("5. Launch dashboard: streamlit run streamlit_app.py")
        print("6. Deploy to cloud: ./scripts/deploy.sh")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
