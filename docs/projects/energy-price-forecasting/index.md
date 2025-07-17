# Energy Price Forecasting - MLOps Pipeline

Production-ready time series forecasting for European electricity markets with FastAPI, Streamlit, and Google Cloud deployment.

---

## 🎯 **What This Project Does**

- ⚡ **Price Forecasting** for German, French, and Dutch electricity markets (up to 168 hours)
- 🤖 **Multiple ML Models** including XGBoost, LightGBM, ARIMA, and ensemble methods
- 🚀 **Production API** with FastAPI for real-time predictions
- 📈 **Interactive Dashboard** built with Streamlit
- ☁️ **Cloud Deployment** on Google Cloud Platform with auto-scaling

---

## 🌐 **Live Demo**

**🎬 Demo Video**: [Watch System in Action](https://youtu.be/ZK0IV5H3RXo)

**📚 API Documentation**: [FastAPI Swagger UI](https://energy-price-forecasting-kj657erbda-uc.a.run.app/docs)

---

## 🚀 **How to Use**

### **API Usage**

**Get Forecast:**
```bash
curl -X POST "https://energy-price-forecasting-kj657erbda-uc.a.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"country": "DE", "model": "xgboost"}'
```

**Health Check:**
```bash
curl https://energy-price-forecasting-kj657erbda-uc.a.run.app/health
```

### **Local Development**

```bash
# Clone and setup
git clone https://github.com/aishwaryaj7/aishwaryaj7.github.io.git
cd aishwaryaj7.github.io/ai-portfolio/projects/energy_price_forecasting

# Install dependencies
pip install -r requirements.txt

# Train models and start API
python train.py
uvicorn api.main:app --reload

# Launch dashboard (separate terminal)
streamlit run streamlit_app.py
```

---

## 🔧 **Key Features**

- **Multi-Market Support**: Germany, France, Netherlands
- **Flexible Forecasting**: 1-168 hour prediction horizons
- **Multiple Models**: XGBoost, LightGBM, ARIMA, Ensemble
- **Real-time API**: FastAPI with automatic documentation
- **Interactive Dashboard**: Streamlit visualization
- **Cloud Deployment**: Google Cloud Run with Docker
- **MLOps Integration**: MLflow for experiment tracking

---

## 🛠️ **Tech Stack**

**Machine Learning**: Scikit-learn, XGBoost, LightGBM, Statsmodels
**API**: FastAPI, Uvicorn, Pydantic
**Frontend**: Streamlit
**Cloud**: Google Cloud Run, Docker
**MLOps**: MLflow
**CI/CD**: GitHub Actions

---

## 🤝 **Skills Demonstrated**

- Time Series Forecasting & MLOps
- FastAPI Development & Cloud Deployment
- Docker Containerization & CI/CD
- Google Cloud Platform & Auto-scaling
- Data Engineering & API Design

---
