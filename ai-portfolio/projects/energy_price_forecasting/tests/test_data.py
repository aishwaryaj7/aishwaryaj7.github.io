"""
Tests for data processing modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.extractors import ENTSOEExtractor, WeatherExtractor
from src.data.transformers import EnergyDataTransformer
from src.data.loaders import GCSDataLoader
from src.utils.helpers import validate_data_quality, create_time_features

class TestDataExtractors:
    """Test data extraction functionality."""
    
    def test_entso_extractor_initialization(self):
        """Test ENTSO-E extractor initialization."""
        extractor = ENTSOEExtractor("dummy_token")
        assert extractor.api_token == "dummy_token"
        assert "DE" in extractor.domain_mappings
        assert "FR" in extractor.domain_mappings
        assert "NL" in extractor.domain_mappings
    
    def test_weather_extractor_initialization(self):
        """Test weather extractor initialization."""
        extractor = WeatherExtractor("dummy_key")
        assert extractor.api_key == "dummy_key"
        assert "DE" in extractor.city_coords
        assert "FR" in extractor.city_coords
        assert "NL" in extractor.city_coords
    
    @pytest.mark.asyncio
    async def test_synthetic_weather_data_generation(self):
        """Test synthetic weather data generation."""
        extractor = WeatherExtractor("dummy_key")
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        weather_data = await extractor.get_historical_weather("DE", start_date, end_date)
        
        assert not weather_data.empty
        assert "temperature" in weather_data.columns
        assert "wind_speed" in weather_data.columns
        assert "cloud_cover" in weather_data.columns
        assert weather_data["country"].iloc[0] == "DE"

class TestDataTransformers:
    """Test data transformation functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.transformer = EnergyDataTransformer()
        
        # Create sample data
        date_range = pd.date_range(start="2024-01-01", end="2024-01-03", freq="H")
        self.sample_price_data = pd.DataFrame({
            'datetime': date_range,
            'price': np.random.normal(50, 10, len(date_range)),
            'country': 'DE',
            'currency': 'EUR/MWh'
        })
        
        self.sample_load_data = pd.DataFrame({
            'datetime': date_range,
            'load': np.random.normal(40000, 5000, len(date_range)),
            'country': 'DE',
            'unit': 'MW'
        })
    
    def test_transformer_initialization(self):
        """Test transformer initialization."""
        transformer = EnergyDataTransformer()
        assert transformer.scalers == {}
        assert transformer.imputers == {}
        assert transformer.target_column == 'price'
    
    def test_clean_price_data(self):
        """Test price data cleaning."""
        self.setUp()
        
        # Add some problematic data
        dirty_data = self.sample_price_data.copy()
        dirty_data.loc[0, 'price'] = -600  # Extreme negative price
        dirty_data.loc[1, 'price'] = 1500  # Extreme high price
        
        cleaned_data = self.transformer._clean_price_data(dirty_data)
        
        assert not cleaned_data.empty
        assert pd.isna(cleaned_data.loc[1, 'price'])  # Extreme high should be NaN
        assert cleaned_data.loc[0, 'price'] == -600  # Negative prices can be valid
    
    def test_merge_datasets(self):
        """Test dataset merging."""
        self.setUp()
        
        merged_data = self.transformer._merge_datasets(
            self.sample_price_data,
            self.sample_load_data,
            pd.DataFrame(),  # Empty renewable data
            pd.DataFrame()   # Empty weather data
        )
        
        assert not merged_data.empty
        assert 'price' in merged_data.columns
        assert 'load' in merged_data.columns
        assert len(merged_data) == len(self.sample_price_data)

class TestDataLoaders:
    """Test data loading functionality."""
    
    def test_gcs_loader_initialization(self):
        """Test GCS loader initialization."""
        loader = GCSDataLoader()
        assert loader.bucket_name is not None
        # Client might be None if no credentials available
    
    def test_local_fallback_save(self):
        """Test local fallback for data saving."""
        loader = GCSDataLoader()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='H'),
            'price': np.random.normal(50, 10, 10),
            'country': 'DE'
        })
        
        # This should fallback to local storage
        saved_path = loader._save_locally(sample_data, "test/sample_data.parquet")
        
        assert "test/sample_data.parquet" in saved_path
        
        # Clean up
        import os
        if os.path.exists(saved_path):
            os.remove(saved_path)

class TestHelperFunctions:
    """Test utility helper functions."""
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Create test data with some quality issues
        test_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='H'),
            'price': np.random.normal(50, 10, 100),
            'country': 'DE'
        })
        
        # Add some missing values
        test_data.loc[0:5, 'price'] = np.nan
        
        quality_report = validate_data_quality(test_data, ['datetime', 'country'])
        
        assert quality_report['total_rows'] == 100
        assert quality_report['missing_values']['price'] == 6
        assert 0 <= quality_report['quality_score'] <= 1
    
    def test_create_time_features(self):
        """Test time feature creation."""
        test_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=48, freq='H'),
            'price': np.random.normal(50, 10, 48)
        })
        
        featured_data = create_time_features(test_data, 'datetime')
        
        assert 'hour' in featured_data.columns
        assert 'day_of_week' in featured_data.columns
        assert 'month' in featured_data.columns
        assert 'hour_sin' in featured_data.columns
        assert 'hour_cos' in featured_data.columns
        assert 'is_weekend' in featured_data.columns
        
        # Check hour values are correct
        assert featured_data['hour'].min() >= 0
        assert featured_data['hour'].max() <= 23

class TestDataIntegration:
    """Integration tests for data pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline integration."""
        # Initialize components
        transformer = EnergyDataTransformer()
        
        # Create synthetic data for all sources
        date_range = pd.date_range(start="2024-01-01", periods=72, freq="H")
        
        price_data = pd.DataFrame({
            'datetime': date_range,
            'price': np.random.normal(50, 10, len(date_range)),
            'country': 'DE',
            'currency': 'EUR/MWh'
        })
        
        load_data = pd.DataFrame({
            'datetime': date_range,
            'load': np.random.normal(40000, 5000, len(date_range)),
            'country': 'DE',
            'unit': 'MW'
        })
        
        renewable_data = pd.DataFrame({
            'datetime': date_range,
            'renewable_generation': np.random.normal(15000, 3000, len(date_range)),
            'country': 'DE',
            'unit': 'MW'
        })
        
        weather_data = pd.DataFrame({
            'datetime': date_range,
            'temperature': np.random.normal(15, 8, len(date_range)),
            'wind_speed': np.random.normal(8, 4, len(date_range)),
            'country': 'DE'
        })
        
        # Run transformation pipeline
        transformed_data = transformer.transform_raw_data(
            price_data, load_data, renewable_data, weather_data
        )
        
        # Verify results
        assert not transformed_data.empty
        assert 'price' in transformed_data.columns
        assert 'load' in transformed_data.columns
        assert 'renewable_generation' in transformed_data.columns
        assert 'temperature' in transformed_data.columns
        
        # Prepare for modeling
        final_data, feature_cols = transformer.prepare_for_modeling(transformed_data)
        
        assert not final_data.empty
        assert len(feature_cols) > 0
        assert 'price' in final_data.columns

if __name__ == "__main__":
    pytest.main([__file__])
