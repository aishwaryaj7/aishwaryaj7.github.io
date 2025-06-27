# ⚡ Energy Price Forecasting

An end-to-end MLOps project for predicting intraday electricity prices and detecting arbitrage opportunities across European energy markets.

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline for forecasting electricity prices in the EPEX Spot markets (Germany, France, Netherlands). It combines time series forecasting with modern MLOps practices to deliver real-time predictions and trading insights.

### Key Features

- **Multi-source Data Integration**: ENTSO-E Transparency Platform, OpenWeatherMap API
- **Advanced ML Models**: XGBoost, LightGBM, ARIMA, and ensemble methods
- **Real-time API**: FastAPI service for price predictions
- **Interactive Dashboard**: Streamlit app for visualization and monitoring
- **Cloud-native**: Deployed on Google Cloud Platform with CI/CD
- **MLOps Pipeline**: Model versioning, experiment tracking with MLflow

## 🎬 Demo

[![Energy Price Forecasting Demo](https://img.youtube.com/vi/ZK0IV5H3RXo/hqdefault.jpg)](https://youtu.be/ZK0IV5H3RXo)

**Watch the full demo**: [Energy Price Forecasting System in Action](https://youtu.be/ZK0IV5H3RXo)


**📚 API Documentation**: [ https://energy-price-forecasting-kj657erbda-uc.a.run.app/docs](https://energy-price-forecasting-kj657erbda-uc.a.run.app/docs)

## 🏗️ Architecture

```
Data Sources → Data Pipeline → ML Models → API Service → Dashboard
     ↓              ↓            ↓          ↓           ↓
  ENTSO-E      GCS Storage   MLflow    Cloud Run   Streamlit
OpenWeather   Transformation  Tracking   FastAPI    Dashboard
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker
- Google Cloud SDK
- Git

## 📊 Data Sources

- **ENTSO-E Transparency Platform**: Day-ahead prices, actual load, renewable generation
- **OpenWeatherMap**: Weather data (temperature, wind speed, cloud cover)
- **Market Coverage**: Germany (DE), France (FR), Netherlands (NL)

## 🤖 Models

### Baseline Models
- **Naive Forecasting**: Simple persistence and seasonal naive
- **ARIMA**: Seasonal ARIMA with automatic parameter selection

### Advanced Models
- **XGBoost**: Gradient boosting with time series features
- **LightGBM**: Fast gradient boosting with categorical features
- **Ensemble**: Weighted combination of multiple models

### Features
- Lagged prices (1h, 24h, 168h)
- Rolling statistics (mean, std, min, max)
- Time-based features (hour, day, month, seasonality)
- Weather variables
- Load and renewable generation

## 📈 Dashboard Features

- **Real-time Forecasting**: Interactive price predictions
- **Market Analytics**: Price patterns and comparisons
- **Arbitrage Detection**: Cross-border trading opportunities
- **Model Performance**: Evaluation metrics and feature importance

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
- Code quality checks (linting, formatting)
- Automated testing
- Docker image building
- Cloud deployment
- Model training and evaluation

## 📁 Project Structure

```
energy_price_forecasting/
├── src/                    # Source code
│   ├── data/              # Data extraction and processing
│   ├── models/            # ML models and evaluation
│   ├── analysis/          # EDA and analytics
│   └── utils/             # Utilities and configuration
├── api/                   # FastAPI application
├── tests/                 # Test suite
├── scripts/               # Deployment scripts
├── notebooks/             # Jupyter notebooks
├── .github/workflows/     # CI/CD configuration
├── train.py              # Training pipeline
├── streamlit_app.py      # Dashboard application
├── Dockerfile            # Multi-stage Docker build
└── requirements.txt      # Python dependencies
```

## 🌟 Key Technologies

- **ML/Data**: pandas, scikit-learn, XGBoost, LightGBM, statsmodels
- **API**: FastAPI, Pydantic, uvicorn
- **Dashboard**: Streamlit, Plotly
- **Cloud**: Google Cloud Run, Cloud Storage, Secret Manager
- **MLOps**: MLflow, Docker, GitHub Actions
- **Data Sources**: ENTSO-E API, OpenWeatherMap API

## 👤 Author

**Aishwarya Jauhari Sharma**

---

*This project demonstrates end-to-end MLOps capabilities for time series forecasting in the energy domain, showcasing modern data science and engineering practices.*
