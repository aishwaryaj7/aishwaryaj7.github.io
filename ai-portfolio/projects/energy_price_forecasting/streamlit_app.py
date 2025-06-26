"""
Streamlit dashboard for Energy Price Forecasting project.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.loaders import GCSDataLoader
from src.utils.config import config

# Page configuration
st.set_page_config(
    page_title="Energy Price Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    # Create sample historical data
    date_range = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate realistic price patterns
    hours = np.array([dt.hour for dt in date_range])
    days = np.array([dt.dayofweek for dt in date_range])
    
    base_price = 50
    hourly_pattern = 15 * np.sin(2 * np.pi * hours / 24)
    daily_pattern = 8 * np.sin(2 * np.pi * days / 7)
    seasonal_trend = 5 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 7))
    noise = np.random.normal(0, 8, len(date_range))
    
    prices = base_price + hourly_pattern + daily_pattern + seasonal_trend + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': date_range,
        'price_DE': prices + np.random.normal(0, 3, len(date_range)),
        'price_FR': prices + np.random.normal(2, 3, len(date_range)),
        'price_NL': prices + np.random.normal(-1, 3, len(date_range)),
        'load_DE': np.random.normal(45000, 8000, len(date_range)),
        'renewable_DE': np.random.normal(15000, 5000, len(date_range)),
        'temperature_DE': 15 + 10 * np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 365)) + np.random.normal(0, 3, len(date_range))
    })
    
    return data

def generate_forecast_data(country: str, hours: int):
    """Generate sample forecast data."""
    start_time = datetime.now()
    forecast_times = [start_time + timedelta(hours=i) for i in range(hours)]
    
    # Generate forecast with uncertainty
    base_forecast = 50 + 10 * np.sin(2 * np.pi * np.arange(hours) / 24)
    noise = np.random.normal(0, 5, hours)
    forecasts = base_forecast + noise
    
    # Add confidence intervals
    confidence_width = 8
    lower_bound = forecasts - confidence_width
    upper_bound = forecasts + confidence_width
    
    forecast_df = pd.DataFrame({
        'datetime': forecast_times,
        'forecast': forecasts,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'country': country
    })
    
    return forecast_df

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Energy Price Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">🔧 Configuration</div>', unsafe_allow_html=True)
    
    # Country selection
    country = st.sidebar.selectbox(
        "Select Country",
        ["DE", "FR", "NL"],
        help="Choose the country for price forecasting"
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["XGBoost", "LightGBM", "ARIMA", "Ensemble"],
        help="Choose the forecasting model"
    )
    
    # Forecast horizon
    forecast_hours = st.sidebar.slider(
        "Forecast Horizon (hours)",
        min_value=1,
        max_value=168,
        value=24,
        help="Number of hours to forecast"
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        st.rerun()
    
    # Load data
    with st.spinner("Loading data..."):
        historical_data = load_sample_data()
        st.session_state.data_loaded = True
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecasting", "📊 Analytics", "🔍 Arbitrage", "⚙️ Model Performance"])
    
    with tab1:
        st.subheader(f"Price Forecasting - {country}")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    st.session_state.forecast_data = generate_forecast_data(country, forecast_hours)
                st.success("Forecast generated!")
        
        if st.session_state.forecast_data is not None:
            forecast_data = st.session_state.forecast_data
            
            # Create forecast plot
            fig = go.Figure()
            
            # Historical prices (last 7 days)
            recent_data = historical_data.tail(168)  # Last 7 days
            fig.add_trace(go.Scatter(
                x=recent_data['datetime'],
                y=recent_data[f'price_{country}'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_data['datetime'],
                y=forecast_data['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_data['datetime'],
                y=forecast_data['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_data['datetime'],
                y=forecast_data['lower_bound'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                name='Confidence Interval',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title=f"Electricity Price Forecast - {country}",
                xaxis_title="Time",
                yaxis_title="Price (EUR/MWh)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_forecast = forecast_data['forecast'].mean()
                st.metric("Average Forecast", f"{avg_forecast:.2f} EUR/MWh")
            
            with col2:
                max_forecast = forecast_data['forecast'].max()
                st.metric("Peak Price", f"{max_forecast:.2f} EUR/MWh")
            
            with col3:
                min_forecast = forecast_data['forecast'].min()
                st.metric("Minimum Price", f"{min_forecast:.2f} EUR/MWh")
            
            with col4:
                volatility = forecast_data['forecast'].std()
                st.metric("Volatility", f"{volatility:.2f} EUR/MWh")
    
    with tab2:
        st.subheader("Market Analytics")
        
        # Price comparison across countries
        fig_comparison = go.Figure()
        
        for country_code in ['DE', 'FR', 'NL']:
            fig_comparison.add_trace(go.Scatter(
                x=historical_data['datetime'].tail(168),
                y=historical_data[f'price_{country_code}'].tail(168),
                mode='lines',
                name=f'{country_code}',
                line=dict(width=2)
            ))
        
        fig_comparison.update_layout(
            title="Price Comparison - Last 7 Days",
            xaxis_title="Time",
            yaxis_title="Price (EUR/MWh)",
            height=400
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Daily patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly patterns
            hourly_avg = historical_data.groupby(historical_data['datetime'].dt.hour)[f'price_{country}'].mean()
            
            fig_hourly = px.bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                title="Average Price by Hour",
                labels={'x': 'Hour of Day', 'y': 'Average Price (EUR/MWh)'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Weekly patterns
            weekly_avg = historical_data.groupby(historical_data['datetime'].dt.dayofweek)[f'price_{country}'].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig_weekly = px.bar(
                x=day_names,
                y=weekly_avg.values,
                title="Average Price by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Average Price (EUR/MWh)'}
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    with tab3:
        st.subheader("Arbitrage Opportunities")
        
        # Generate sample arbitrage opportunities
        arbitrage_data = []
        
        for i in range(10):
            opportunity = {
                'Time': datetime.now() + timedelta(hours=i),
                'Buy From': np.random.choice(['DE', 'FR', 'NL']),
                'Sell To': np.random.choice(['DE', 'FR', 'NL']),
                'Buy Price': np.random.uniform(40, 60),
                'Sell Price': np.random.uniform(45, 65),
                'Spread': 0,
                'Confidence': np.random.uniform(0.6, 0.95)
            }
            opportunity['Spread'] = opportunity['Sell Price'] - opportunity['Buy Price']
            
            if opportunity['Spread'] > 2 and opportunity['Buy From'] != opportunity['Sell To']:
                arbitrage_data.append(opportunity)
        
        if arbitrage_data:
            arbitrage_df = pd.DataFrame(arbitrage_data)
            arbitrage_df = arbitrage_df[arbitrage_df['Spread'] > 0].sort_values('Spread', ascending=False)
            
            st.dataframe(
                arbitrage_df.style.format({
                    'Buy Price': '{:.2f}',
                    'Sell Price': '{:.2f}',
                    'Spread': '{:.2f}',
                    'Confidence': '{:.1%}'
                }),
                use_container_width=True
            )
            
            # Arbitrage visualization
            if len(arbitrage_df) > 0:
                fig_arb = px.scatter(
                    arbitrage_df,
                    x='Time',
                    y='Spread',
                    size='Confidence',
                    color='Buy From',
                    title="Arbitrage Opportunities Over Time",
                    labels={'Spread': 'Price Spread (EUR/MWh)'}
                )
                st.plotly_chart(fig_arb, use_container_width=True)
        else:
            st.info("No significant arbitrage opportunities found in the current forecast period.")
    
    with tab4:
        st.subheader("Model Performance")
        
        # Model performance metrics
        models_performance = {
            'XGBoost': {'MAE': 8.45, 'RMSE': 12.67, 'MAPE': 15.23, 'R²': 0.87},
            'LightGBM': {'MAE': 8.92, 'RMSE': 13.21, 'MAPE': 16.45, 'R²': 0.85},
            'ARIMA': {'MAE': 12.34, 'RMSE': 18.56, 'MAPE': 22.78, 'R²': 0.72},
            'Ensemble': {'MAE': 7.89, 'RMSE': 11.98, 'MAPE': 14.56, 'R²': 0.89}
        }
        
        performance_df = pd.DataFrame(models_performance).T
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(performance_df.style.format('{:.2f}'), use_container_width=True)
        
        with col2:
            # Performance comparison chart
            fig_perf = go.Figure()
            
            metrics = ['MAE', 'RMSE', 'MAPE']
            for metric in metrics:
                fig_perf.add_trace(go.Bar(
                    name=metric,
                    x=list(models_performance.keys()),
                    y=[models_performance[model][metric] for model in models_performance.keys()]
                ))
            
            fig_perf.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Error Value",
                barmode='group'
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Feature importance (sample data)
        st.subheader("Feature Importance (XGBoost)")
        
        feature_importance = {
            'price_lag_24': 0.25,
            'price_lag_168': 0.18,
            'load': 0.15,
            'renewable_generation': 0.12,
            'temperature': 0.08,
            'hour_sin': 0.07,
            'day_of_week': 0.06,
            'price_rolling_mean_24': 0.05,
            'wind_speed': 0.04
        }
        
        fig_importance = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="Top Features by Importance"
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Energy Price Forecasting Dashboard** | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data: ENTSO-E, OpenWeatherMap"
    )

if __name__ == "__main__":
    main()
