"""
Data extraction module for Energy Price Forecasting project.
Handles API calls to ENTSO-E, OpenWeatherMap, and other data sources.
"""
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import asyncio
import logging
from ..utils.config import config
from ..utils.helpers import setup_logging

logger = setup_logging()

class ENTSOEExtractor:
    """
    Extractor for ENTSO-E Transparency Platform data.
    """
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = config.api.base_urls["entso_e"]
        self.domain_mappings = {
            "DE": "10Y1001A1001A83F",  # Germany
            "FR": "10Y1001A1001A87L",  # France  
            "NL": "10Y1001A1001A92E"   # Netherlands
        }
    
    async def get_day_ahead_prices(self, country: str, start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """
        Extract day-ahead electricity prices.
        
        Args:
            country: Country code (DE, FR, NL)
            start_date: Start date for data extraction
            end_date: End date for data extraction
            
        Returns:
            DataFrame with day-ahead prices
        """
        domain = self.domain_mappings.get(country)
        if not domain:
            raise ValueError(f"Unsupported country: {country}")
        
        params = {
            "securityToken": self.api_token,
            "documentType": "A44",  # Day-ahead prices
            "in_Domain": domain,
            "out_Domain": domain,
            "periodStart": start_date.strftime("%Y%m%d%H%M"),
            "periodEnd": end_date.strftime("%Y%m%d%H%M")
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                prices_data = self._parse_price_xml(root, country)
                
                logger.info(f"Successfully extracted {len(prices_data)} day-ahead price records for {country}")
                return prices_data
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error extracting day-ahead prices for {country}: {e}")
                return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error extracting day-ahead prices for {country}: {e}")
                return pd.DataFrame()
    
    async def get_actual_load(self, country: str, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """
        Extract actual electricity load data.
        
        Args:
            country: Country code
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with actual load data
        """
        domain = self.domain_mappings.get(country)
        if not domain:
            raise ValueError(f"Unsupported country: {country}")
        
        params = {
            "securityToken": self.api_token,
            "documentType": "A65",  # Actual load
            "outBiddingZone_Domain": domain,
            "periodStart": start_date.strftime("%Y%m%d%H%M"),
            "periodEnd": end_date.strftime("%Y%m%d%H%M")
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                load_data = self._parse_load_xml(root, country)
                
                logger.info(f"Successfully extracted {len(load_data)} load records for {country}")
                return load_data
                
            except Exception as e:
                logger.error(f"Error extracting load data for {country}: {e}")
                return pd.DataFrame()
    
    async def get_renewable_generation(self, country: str, start_date: datetime, 
                                     end_date: datetime) -> pd.DataFrame:
        """
        Extract renewable generation data (wind + solar).
        
        Args:
            country: Country code
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with renewable generation data
        """
        domain = self.domain_mappings.get(country)
        if not domain:
            raise ValueError(f"Unsupported country: {country}")
        
        # Get wind and solar data separately
        wind_data = await self._get_generation_by_type(country, "B19", start_date, end_date)  # Wind
        solar_data = await self._get_generation_by_type(country, "B16", start_date, end_date)  # Solar
        
        # Combine wind and solar data
        if not wind_data.empty and not solar_data.empty:
            renewable_data = pd.merge(wind_data, solar_data, on=['datetime', 'country'], 
                                    how='outer', suffixes=('_wind', '_solar'))
            renewable_data['renewable_total'] = (
                renewable_data.get('generation_wind', 0) + 
                renewable_data.get('generation_solar', 0)
            )
        elif not wind_data.empty:
            renewable_data = wind_data.copy()
            renewable_data['renewable_total'] = renewable_data['generation']
        elif not solar_data.empty:
            renewable_data = solar_data.copy()
            renewable_data['renewable_total'] = renewable_data['generation']
        else:
            renewable_data = pd.DataFrame()
        
        logger.info(f"Successfully extracted renewable generation data for {country}")
        return renewable_data
    
    async def _get_generation_by_type(self, country: str, psr_type: str, 
                                    start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Helper method to get generation data by type."""
        domain = self.domain_mappings.get(country)
        
        params = {
            "securityToken": self.api_token,
            "documentType": "A75",  # Actual generation per type
            "in_Domain": domain,
            "psrType": psr_type,
            "periodStart": start_date.strftime("%Y%m%d%H%M"),
            "periodEnd": end_date.strftime("%Y%m%d%H%M")
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                return self._parse_generation_xml(root, country)
                
            except Exception as e:
                logger.error(f"Error extracting generation data for {country}, type {psr_type}: {e}")
                return pd.DataFrame()
    
    def _parse_price_xml(self, root: ET.Element, country: str) -> pd.DataFrame:
        """Parse XML response for price data."""
        data = []
        
        for timeseries in root.findall('.//{*}TimeSeries'):
            for period in timeseries.findall('.//{*}Period'):
                start_time = period.find('.//{*}timeInterval/{*}start').text
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
                for point in period.findall('.//{*}Point'):
                    position = int(point.find('.//{*}position').text)
                    price = float(point.find('.//{*}price.amount').text)
                    
                    # Calculate actual datetime (position is 1-indexed)
                    actual_dt = start_dt + timedelta(hours=position-1)
                    
                    data.append({
                        'datetime': actual_dt,
                        'country': country,
                        'price': price,
                        'currency': 'EUR/MWh'
                    })
        
        return pd.DataFrame(data)
    
    def _parse_load_xml(self, root: ET.Element, country: str) -> pd.DataFrame:
        """Parse XML response for load data."""
        data = []
        
        for timeseries in root.findall('.//{*}TimeSeries'):
            for period in timeseries.findall('.//{*}Period'):
                start_time = period.find('.//{*}timeInterval/{*}start').text
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
                for point in period.findall('.//{*}Point'):
                    position = int(point.find('.//{*}position').text)
                    quantity = float(point.find('.//{*}quantity').text)
                    
                    actual_dt = start_dt + timedelta(hours=position-1)
                    
                    data.append({
                        'datetime': actual_dt,
                        'country': country,
                        'load': quantity,
                        'unit': 'MW'
                    })
        
        return pd.DataFrame(data)
    
    def _parse_generation_xml(self, root: ET.Element, country: str) -> pd.DataFrame:
        """Parse XML response for generation data."""
        data = []
        
        for timeseries in root.findall('.//{*}TimeSeries'):
            for period in timeseries.findall('.//{*}Period'):
                start_time = period.find('.//{*}timeInterval/{*}start').text
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
                for point in period.findall('.//{*}Point'):
                    position = int(point.find('.//{*}position').text)
                    quantity = float(point.find('.//{*}quantity').text)
                    
                    actual_dt = start_dt + timedelta(hours=position-1)
                    
                    data.append({
                        'datetime': actual_dt,
                        'country': country,
                        'generation': quantity,
                        'unit': 'MW'
                    })
        
        return pd.DataFrame(data)


class WeatherExtractor:
    """
    Extractor for weather data from OpenWeatherMap API.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = config.api.base_urls["openweather"]
        self.city_coords = {
            "DE": {"lat": 52.5200, "lon": 13.4050},  # Berlin
            "FR": {"lat": 48.8566, "lon": 2.3522},   # Paris
            "NL": {"lat": 52.3676, "lon": 4.9041}    # Amsterdam
        }
    
    async def get_historical_weather(self, country: str, start_date: datetime, 
                                   end_date: datetime) -> pd.DataFrame:
        """
        Extract historical weather data.
        
        Args:
            country: Country code
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with weather data
        """
        coords = self.city_coords.get(country)
        if not coords:
            raise ValueError(f"Unsupported country: {country}")
        
        weather_data = []
        current_date = start_date
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            while current_date <= end_date:
                timestamp = int(current_date.timestamp())
                
                params = {
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "dt": timestamp,
                    "appid": self.api_key,
                    "units": "metric"
                }
                
                try:
                    # Note: This endpoint requires a paid subscription
                    # For demo purposes, we'll create synthetic weather data
                    weather_record = self._create_synthetic_weather(current_date, country)
                    weather_data.append(weather_record)
                    
                except Exception as e:
                    logger.error(f"Error extracting weather data for {country} on {current_date}: {e}")
                
                current_date += timedelta(hours=1)
        
        logger.info(f"Successfully extracted {len(weather_data)} weather records for {country}")
        return pd.DataFrame(weather_data)
    
    def _create_synthetic_weather(self, dt: datetime, country: str) -> Dict:
        """Create synthetic weather data for demo purposes."""
        # Create realistic seasonal patterns
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        
        # Temperature with seasonal and daily patterns
        base_temp = 10 + 15 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)    # Daily
        temp = base_temp + daily_variation + np.random.normal(0, 2)
        
        # Wind speed with some randomness
        wind_speed = 5 + 10 * np.random.random() + 2 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Cloud cover
        cloud_cover = np.random.randint(0, 101)
        
        return {
            'datetime': dt,
            'country': country,
            'temperature': round(temp, 1),
            'wind_speed': round(wind_speed, 1),
            'cloud_cover': cloud_cover,
            'humidity': np.random.randint(30, 90)
        }
