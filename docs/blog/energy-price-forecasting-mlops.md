# Building an End-to-End MLOps Pipeline: Predicting Electricity Prices

**By Aishwarya Jauhari Sharma**  
*Published: June 2025*

---

## 🎯 **Introduction**

A few weeks ago, I decided to challenge myself with a project that would combine machine learning, cloud deployment, and real-world data. The result? An end-to-end MLOps pipeline that predicts electricity prices across European markets. Here's my journey and the key concepts I discovered that every aspiring ML engineer should know.

## 🚀 **The Problem That Got Me Excited**

Energy trading is fascinating. Electricity prices change every hour based on demand, weather, renewable generation, and market dynamics. I wondered: could I build a system that predicts these prices and helps identify profitable trading opportunities?

The challenge wasn't just building a model – it was creating a complete system that could:
- Fetch real-time data from multiple sources
- Generate predictions on demand
- Serve results through a professional API
- Display insights in an interactive dashboard
- Run reliably in the cloud

## 🏗️ **What I Built**

My final system includes:
- **FastAPI backend** that generates price forecasts
- **Streamlit dashboard** for interactive visualization
- **Google Cloud deployment** with auto-scaling
- **Docker containers** for consistent deployment
- **Real-time prediction engine** with multiple model support

You can check out the [live demo](https://youtu.be/ZK0IV5H3RXo) and explore the [API documentation]( https://energy-price-forecasting-kj657erbda-uc.a.run.app/docs).

## 📊 **Understanding Time Series Forecasting in Energy Markets**

Working with electricity price data taught me fascinating patterns about energy markets. Electricity prices follow predictable cycles - they're higher during business hours when demand peaks, lower on weekends, and influenced by weather patterns that affect both demand (heating/cooling) and supply (renewable generation).

The forecasting engine I built captures these patterns using multiple approaches:

```python
def generate_price_forecast(country, model_type, forecast_horizon):
    # Different countries have different market characteristics
    market_params = get_market_parameters(country)
    
    predictions = []
    for hour in range(forecast_horizon):
        # Capture daily demand cycles
        daily_pattern = calculate_daily_demand_cycle(hour)
        
        # Account for weekly patterns (weekends vs weekdays)
        weekly_pattern = calculate_weekly_pattern(hour)
        
        # Apply model-specific forecasting logic
        price = apply_model_prediction(
            daily_pattern, weekly_pattern, market_params, model_type
        )
        
        predictions.append({
            "datetime": get_forecast_time(hour),
            "predicted_price": price,
            "confidence_interval": calculate_confidence(price, model_type)
        })
    
    return predictions
```

What I learned about time series forecasting:
    - **Seasonality matters**: Energy prices have multiple seasonal patterns (daily, weekly, yearly)
    - **External factors**: Weather, holidays, and economic events significantly impact prices
    - **Confidence intervals**: In volatile markets, knowing the uncertainty is as important as the prediction itself

## 🔧 **Building Production APIs: Why FastAPI Changed My Perspective**

I chose FastAPI for the backend, and it completely changed how I think about API development. The framework embodies several important concepts that every developer should understand:

```python
@app.post("/predict")
def predict(request: ForecastRequest):
    # Validate inputs
    if request.country not in ["DE", "FR", "NL"]:
        raise HTTPException(status_code=400, detail="Invalid country")
    
    # Generate predictions
    predictions = generate_realistic_forecast(
        request.country, 
        request.model_name, 
        request.forecast_horizon
    )
    
    return {
        "predictions": predictions,
        "model_used": request.model_name,
        "country": request.country,
        "generated_at": datetime.now()
    }
```

### Key API Design Concepts I Learned:

**1. Contract-First Development**: By defining Pydantic models first, you establish a clear contract between your API and its consumers. This prevents many integration issues down the line.

**2. Automatic Documentation**: FastAPI generates interactive documentation automatically. Visit `/docs` and you get a beautiful, testable API explorer. This taught me that documentation isn't separate from code - it should be part of it.

**3. Type Safety**: Python's type hints aren't just for readability. FastAPI uses them for validation, serialization, and documentation generation. This is a powerful example of how modern Python leverages static typing.

**4. Async by Default**: FastAPI is built for async operations, which is crucial for I/O-heavy applications like ML inference APIs that might call external services or databases.

## 📱 **Interactive Dashboards: The Power of Streamlit**

For the frontend, I discovered Streamlit - a framework that embodies the principle of "simplicity without sacrificing functionality." It lets you build interactive web apps with pure Python, which taught me important lessons about rapid prototyping and user experience design.

```python
# Sidebar controls
country = st.sidebar.selectbox("Select Country", ["DE", "FR", "NL"])
model = st.sidebar.selectbox("Select Model", ["xgboost", "lightgbm", "arima"])
hours = st.sidebar.slider("Forecast Horizon", 1, 48, 24)

# Generate forecast when button is clicked
if st.button("Generate Forecast"):
    # Call our API
    response = requests.post(f"{API_URL}/predict", json={
        "country": country,
        "model_name": model,
        "forecast_horizon": hours
    })
    
    # Create interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['price'],
        mode='lines+markers',
        name='Predicted Price'
    ))
    
    st.plotly_chart(fig)
```

### What Streamlit Taught Me About User Experience:

**1. Reactive Programming**: Streamlit follows a reactive model - when users change inputs, the entire app reruns. This might seem inefficient, but it makes the code incredibly simple and predictable.

**2. State Management**: Unlike traditional web frameworks, you don't manage state explicitly. Streamlit handles it for you, which reduces complexity but requires thinking differently about data flow.

**3. Rapid Prototyping**: The ability to go from idea to interactive dashboard in minutes is powerful for data science projects. You can focus on the logic rather than the plumbing.

The dashboard responds dynamically to user selections, making it feel like a professional trading tool.

## ☁️ **Serverless Architecture: Understanding Cloud Run**

Deploying to Google Cloud Run introduced me to serverless containers - a concept that bridges traditional containerization with serverless computing. The idea is brilliant: you package your app in a Docker container, and the cloud provider handles everything else – scaling, load balancing, even scaling to zero when no one's using it.

### Key Serverless Concepts I Discovered:

**1. Stateless Design**: Serverless applications must be stateless because instances can be created and destroyed at any time. This forces you to design better, more resilient applications.

**2. Cold Starts vs. Warm Instances**: When your app hasn't been used for a while, the first request takes longer (cold start). Understanding this helps you optimize for performance.

**3. Environment Variables**: Cloud platforms use environment variables for configuration. Your app needs to be flexible enough to adapt to different environments:

```dockerfile
# Container must listen on the port provided by the platform
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT
```

**4. Container Optimization**: Smaller containers start faster. This taught me to be mindful of dependencies and use multi-stage builds when necessary.

**5. Horizontal Scaling**: Instead of making one big server, serverless creates many small instances. Your code must handle this gracefully.

### Docker Best Practices I Learned:

- **Use specific base images**: `python:3.9-slim` instead of `python:latest`
- **Minimize layers**: Combine RUN commands to reduce image size
- **Don't run as root**: Create a user for security
- **Use .dockerignore**: Exclude unnecessary files from the build context

```python
# Example of clean Pydantic model design
class ForecastRequest(BaseModel):
    country: str = "DE"
    model_name: str = "xgboost"
    forecast_horizon: int = 24
    
    class Config:
        # Avoid using Python built-in names as field names
        json_schema_extra = {
            "example": {
                "country": "DE",
                "model_name": "xgboost",
                "forecast_horizon": 24
            }
        }
```

## 🔄 **Core MLOps Concepts That Changed My Thinking**

This project taught me that MLOps is fundamentally about bridging the gap between data science experimentation and production software engineering. Here are the key concepts that every ML practitioner should understand:

### 1. The Model is Just One Component
In traditional data science, the model is everything. In MLOps, the model is just one piece of a larger system that includes data pipelines, APIs, monitoring, and user interfaces.

### 2. Reproducibility Through Infrastructure as Code
Every aspect of your system should be reproducible. This means:
- **Containerization**: Your app runs the same everywhere
- **Version control**: Not just for code, but for data and models too
- **Environment management**: Dependencies should be explicit and locked

### 3. API-First Design Philosophy
Building the API first forces you to think about:
- **Contract definition**: What inputs and outputs does your system need?
- **Error handling**: How do you gracefully handle edge cases?
- **Documentation**: How do consumers understand your system?

### 4. Observability from Day One
Production systems need monitoring, logging, and alerting. This isn't an afterthought - it's a core requirement.

### 5. The Importance of Feedback Loops
MLOps systems should be designed to learn and improve over time through:
- **Performance monitoring**: Is the model still accurate?
- **Data drift detection**: Is the input data changing?
- **A/B testing**: Which model version performs better?

## 📚 **Key Principles for Building Production ML Systems**

Through this project, I discovered several principles that apply to any ML system:

### 1. Design for Failure
Assume external APIs will be down, models will drift, and containers will crash. Build resilience into your system from the start.

### 2. Make Everything Observable
You can't improve what you can't measure. Instrument your system with metrics, logs, and health checks.

### 3. Embrace Simplicity
Complex systems are hard to debug, deploy, and maintain. Start simple and add complexity only when necessary.

### 4. Think in Terms of Contracts
APIs, data schemas, and model interfaces are contracts. Design them carefully because changing them later is expensive.

### 5. Automate Everything
Manual processes don't scale and are error-prone. Automate testing, deployment, and monitoring from the beginning.


## 💡 **Try It Yourself**

The best way to learn is by doing. If you're interested in MLOps, I'd recommend starting with a simple project. Pick a problem you're curious about, build a basic model, wrap it in an API, and deploy it to the cloud.

The tools are more accessible than ever, and the learning experience is invaluable.

