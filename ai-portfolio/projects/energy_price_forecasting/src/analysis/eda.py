"""
Exploratory Data Analysis module for Energy Price Forecasting project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

from ..utils.helpers import setup_logging, detect_outliers

logger = setup_logging()

class EnergyDataAnalyzer:
    """
    Comprehensive EDA for energy price forecasting data.
    """
    
    def __init__(self):
        self.data = None
        self.analysis_results = {}
    
    def load_data(self, data: pd.DataFrame):
        """Load data for analysis."""
        self.data = data.copy()
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        logger.info(f"Loaded data with shape: {self.data.shape}")
    
    def basic_statistics(self) -> Dict:
        """
        Generate basic statistical summary of the data.
        
        Returns:
            Dictionary with basic statistics
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        stats = {
            'shape': self.data.shape,
            'date_range': {
                'start': self.data['datetime'].min(),
                'end': self.data['datetime'].max(),
                'duration_days': (self.data['datetime'].max() - self.data['datetime'].min()).days
            },
            'missing_values': self.data.isnull().sum().to_dict(),
            'descriptive_stats': self.data[numeric_cols].describe().to_dict(),
            'countries': self.data['country'].unique().tolist() if 'country' in self.data.columns else [],
            'data_frequency': self._detect_frequency()
        }
        
        self.analysis_results['basic_stats'] = stats
        logger.info(f"Data spans {stats['date_range']['duration_days']} days")
        logger.info(f"Countries: {stats['countries']}")
        
        return stats
    
    def _detect_frequency(self) -> str:
        """Detect the frequency of the time series."""
        if len(self.data) < 2:
            return "unknown"
        
        time_diffs = self.data['datetime'].diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        
        if most_common_diff == pd.Timedelta(hours=1):
            return "hourly"
        elif most_common_diff == pd.Timedelta(days=1):
            return "daily"
        elif most_common_diff == pd.Timedelta(minutes=15):
            return "15min"
        else:
            return f"custom_{most_common_diff}"
    
    def price_analysis(self, price_col: str = 'price') -> Dict:
        """
        Analyze electricity price patterns.
        
        Args:
            price_col: Name of the price column
            
        Returns:
            Dictionary with price analysis results
        """
        if price_col not in self.data.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        
        prices = self.data[price_col].dropna()
        
        # Basic price statistics
        price_stats = {
            'mean': prices.mean(),
            'median': prices.median(),
            'std': prices.std(),
            'min': prices.min(),
            'max': prices.max(),
            'negative_prices': (prices < 0).sum(),
            'negative_price_pct': (prices < 0).mean() * 100,
            'extreme_high_prices': (prices > 1000).sum(),  # >1000 EUR/MWh
            'price_volatility': prices.std() / prices.mean() if prices.mean() != 0 else np.inf
        }
        
        # Price spikes and dips detection
        outliers_iqr = detect_outliers(prices, method='iqr', threshold=1.5)
        outliers_zscore = detect_outliers(prices, method='zscore', threshold=3)
        
        price_stats['outliers_iqr'] = outliers_iqr.sum()
        price_stats['outliers_zscore'] = outliers_zscore.sum()
        
        # Hourly patterns
        if 'datetime' in self.data.columns:
            self.data['hour'] = self.data['datetime'].dt.hour
            self.data['day_of_week'] = self.data['datetime'].dt.dayofweek
            self.data['month'] = self.data['datetime'].dt.month
            
            hourly_stats = self.data.groupby('hour')[price_col].agg(['mean', 'std', 'min', 'max'])
            daily_stats = self.data.groupby('day_of_week')[price_col].agg(['mean', 'std', 'min', 'max'])
            monthly_stats = self.data.groupby('month')[price_col].agg(['mean', 'std', 'min', 'max'])
            
            price_stats['hourly_patterns'] = hourly_stats.to_dict()
            price_stats['daily_patterns'] = daily_stats.to_dict()
            price_stats['monthly_patterns'] = monthly_stats.to_dict()
            
            # Peak and off-peak hours
            peak_hours = hourly_stats['mean'].nlargest(6).index.tolist()
            off_peak_hours = hourly_stats['mean'].nsmallest(6).index.tolist()
            
            price_stats['peak_hours'] = peak_hours
            price_stats['off_peak_hours'] = off_peak_hours
        
        self.analysis_results['price_analysis'] = price_stats
        logger.info(f"Average price: {price_stats['mean']:.2f} EUR/MWh")
        logger.info(f"Price volatility: {price_stats['price_volatility']:.2f}")
        
        return price_stats
    
    def seasonal_decomposition(self, price_col: str = 'price', 
                             period: int = 24) -> Dict:
        """
        Perform seasonal decomposition of price series.
        
        Args:
            price_col: Name of the price column
            period: Seasonal period (24 for daily seasonality in hourly data)
            
        Returns:
            Dictionary with decomposition results
        """
        if price_col not in self.data.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        
        # Prepare time series
        ts_data = self.data.set_index('datetime')[price_col].dropna()
        
        if len(ts_data) < 2 * period:
            logger.warning(f"Insufficient data for seasonal decomposition (need at least {2*period} points)")
            return {}
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=period)
            
            decomp_results = {
                'trend': decomposition.trend.dropna(),
                'seasonal': decomposition.seasonal.dropna(),
                'residual': decomposition.resid.dropna(),
                'seasonal_strength': 1 - (decomposition.resid.var() / ts_data.var()),
                'trend_strength': 1 - (decomposition.resid.var() / (ts_data - decomposition.seasonal).var())
            }
            
            self.analysis_results['seasonal_decomposition'] = decomp_results
            logger.info(f"Seasonal strength: {decomp_results['seasonal_strength']:.3f}")
            logger.info(f"Trend strength: {decomp_results['trend_strength']:.3f}")
            
            return decomp_results
            
        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
            return {}
    
    def stationarity_analysis(self, price_col: str = 'price') -> Dict:
        """
        Analyze stationarity of the price series.
        
        Args:
            price_col: Name of the price column
            
        Returns:
            Dictionary with stationarity test results
        """
        if price_col not in self.data.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        
        prices = self.data[price_col].dropna()
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(prices)
        
        stationarity_results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05,
            'differencing_needed': adf_result[1] >= 0.05
        }
        
        # Test first difference if non-stationary
        if not stationarity_results['is_stationary']:
            prices_diff = prices.diff().dropna()
            adf_diff = adfuller(prices_diff)
            
            stationarity_results['first_diff_adf_statistic'] = adf_diff[0]
            stationarity_results['first_diff_adf_pvalue'] = adf_diff[1]
            stationarity_results['first_diff_is_stationary'] = adf_diff[1] < 0.05
        
        self.analysis_results['stationarity'] = stationarity_results
        logger.info(f"Series is {'stationary' if stationarity_results['is_stationary'] else 'non-stationary'}")
        
        return stationarity_results
    
    def correlation_analysis(self) -> Dict:
        """
        Analyze correlations between variables.
        
        Returns:
            Dictionary with correlation analysis results
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return {}
        
        # Correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find high correlations (excluding self-correlations)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        correlation_results = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr_pairs,
            'price_correlations': {}
        }
        
        # Price correlations with other variables
        if 'price' in numeric_cols:
            price_corrs = corr_matrix['price'].drop('price').sort_values(key=abs, ascending=False)
            correlation_results['price_correlations'] = price_corrs.to_dict()
        
        self.analysis_results['correlations'] = correlation_results
        logger.info(f"Found {len(high_corr_pairs)} high correlation pairs")
        
        return correlation_results
    
    def autocorrelation_analysis(self, price_col: str = 'price', 
                               max_lags: int = 48) -> Dict:
        """
        Analyze autocorrelation and partial autocorrelation.
        
        Args:
            price_col: Name of the price column
            max_lags: Maximum number of lags to analyze
            
        Returns:
            Dictionary with autocorrelation results
        """
        if price_col not in self.data.columns:
            raise ValueError(f"Price column '{price_col}' not found")
        
        prices = self.data[price_col].dropna()
        
        if len(prices) <= max_lags:
            max_lags = len(prices) - 1
        
        # Calculate ACF and PACF
        acf_values = acf(prices, nlags=max_lags, fft=True)
        pacf_values = pacf(prices, nlags=max_lags)
        
        # Find significant lags (simplified approach)
        # Approximate 95% confidence interval
        n = len(prices)
        confidence_interval = 1.96 / np.sqrt(n)
        
        significant_acf_lags = np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
        significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > confidence_interval)[0] + 1
        
        autocorr_results = {
            'acf_values': acf_values.tolist(),
            'pacf_values': pacf_values.tolist(),
            'significant_acf_lags': significant_acf_lags.tolist(),
            'significant_pacf_lags': significant_pacf_lags.tolist(),
            'confidence_interval': confidence_interval,
            'max_acf': np.max(np.abs(acf_values[1:])),
            'max_pacf': np.max(np.abs(pacf_values[1:]))
        }
        
        self.analysis_results['autocorrelation'] = autocorr_results
        logger.info(f"Found {len(significant_acf_lags)} significant ACF lags")
        logger.info(f"Found {len(significant_pacf_lags)} significant PACF lags")
        
        return autocorr_results
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            String with formatted summary report
        """
        if not self.analysis_results:
            return "No analysis results available. Run analysis methods first."
        
        report = []
        report.append("=" * 60)
        report.append("ENERGY PRICE FORECASTING - DATA ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Basic statistics
        if 'basic_stats' in self.analysis_results:
            stats = self.analysis_results['basic_stats']
            report.append(f"\nDATA OVERVIEW:")
            report.append(f"- Dataset shape: {stats['shape']}")
            report.append(f"- Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            report.append(f"- Duration: {stats['date_range']['duration_days']} days")
            report.append(f"- Countries: {', '.join(stats['countries'])}")
            report.append(f"- Data frequency: {stats['data_frequency']}")
        
        # Price analysis
        if 'price_analysis' in self.analysis_results:
            price_stats = self.analysis_results['price_analysis']
            report.append(f"\nPRICE ANALYSIS:")
            report.append(f"- Average price: {price_stats['mean']:.2f} EUR/MWh")
            report.append(f"- Price volatility: {price_stats['price_volatility']:.2f}")
            report.append(f"- Negative prices: {price_stats['negative_prices']} ({price_stats['negative_price_pct']:.1f}%)")
            report.append(f"- Extreme high prices (>1000): {price_stats['extreme_high_prices']}")
            
            if 'peak_hours' in price_stats:
                report.append(f"- Peak hours: {price_stats['peak_hours']}")
                report.append(f"- Off-peak hours: {price_stats['off_peak_hours']}")
        
        # Stationarity
        if 'stationarity' in self.analysis_results:
            stationarity = self.analysis_results['stationarity']
            report.append(f"\nSTATIONARITY ANALYSIS:")
            report.append(f"- Series is {'stationary' if stationarity['is_stationary'] else 'non-stationary'}")
            report.append(f"- ADF p-value: {stationarity['adf_pvalue']:.4f}")
            
            if 'first_diff_is_stationary' in stationarity:
                report.append(f"- First difference is {'stationary' if stationarity['first_diff_is_stationary'] else 'non-stationary'}")
        
        # Seasonal decomposition
        if 'seasonal_decomposition' in self.analysis_results:
            seasonal = self.analysis_results['seasonal_decomposition']
            report.append(f"\nSEASONAL ANALYSIS:")
            report.append(f"- Seasonal strength: {seasonal['seasonal_strength']:.3f}")
            report.append(f"- Trend strength: {seasonal['trend_strength']:.3f}")
        
        # Autocorrelation
        if 'autocorrelation' in self.analysis_results:
            autocorr = self.analysis_results['autocorrelation']
            report.append(f"\nAUTOCORRELATION ANALYSIS:")
            report.append(f"- Significant ACF lags: {len(autocorr['significant_acf_lags'])}")
            report.append(f"- Significant PACF lags: {len(autocorr['significant_pacf_lags'])}")
            report.append(f"- Max ACF: {autocorr['max_acf']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
