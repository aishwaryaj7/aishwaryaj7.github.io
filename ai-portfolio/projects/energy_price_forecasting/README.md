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

### Local Development

1. **Clone and setup**:
   ```bash
   cd ai-portfolio/projects/energy_price_forecasting
   pip install -r requirements.txt
   ```

2. **Set up GCP resources**:
   ```bash
   ./scripts/setup_gcp.sh
   ```

3. **Configure environment**:
   ```bash
   cp .env.gcp .env
   # Update API keys in .env file
   ```

4. **Run training pipeline**:
   ```bash
   python train.py
   ```

5. **Start API service**:
   ```bash
   uvicorn api.main:app --reload
   ```

6. **Launch dashboard**:
   ```bash
   streamlit run streamlit_app.py
   ```

### Docker Deployment

```bash
# Build and run API
docker build --target production-api -t energy-api .
docker run -p 8000:8000 energy-api

# Build and run dashboard
docker build --target production-streamlit -t energy-dashboard .
docker run -p 8501:8501 energy-dashboard
```

### Cloud Deployment

```bash
# Deploy to Google Cloud Run
./scripts/deploy.sh
```

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

## 🔧 API Endpoints

- `GET /health` - Health check
- `GET /models/info` - Available models information
- `POST /predict` - Generate price forecasts
- `GET /models/{model}/performance` - Model performance metrics
- `POST /retrain` - Trigger model retraining

## 📈 Dashboard Features

- **Real-time Forecasting**: Interactive price predictions
- **Market Analytics**: Price patterns and comparisons
- **Arbitrage Detection**: Cross-border trading opportunities
- **Model Performance**: Evaluation metrics and feature importance

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_data.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Aishwarya Jauhari**
- Email: aishwarya.jauhari@icloud.com
- LinkedIn: [linkedin.com/in/aishwarya-jauhari](https://linkedin.com/in/aishwarya-jauhari/)
- GitHub: [github.com/aishwaryaj7](https://github.com/aishwaryaj7)

---

*This project demonstrates end-to-end MLOps capabilities for time series forecasting in the energy domain, showcasing modern data science and engineering practices.*
