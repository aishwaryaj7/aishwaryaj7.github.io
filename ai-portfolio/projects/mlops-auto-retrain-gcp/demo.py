#!/usr/bin/env python3
"""
Complete MLOps Demo Script
Demonstrates the entire pipeline: Train -> Evaluate -> Serve -> Test
"""

import subprocess
import time
import requests
import json
import sys
import os
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"⚡ {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def test_api_endpoint(url, data=None, description="API test"):
    """Test an API endpoint."""
    try:
        if data:
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {description} successful")
            return response.json()
        else:
            print(f"❌ {description} failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ {description} failed: Cannot connect to API")
        return None
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return None

def main():
    """Run the complete MLOps demo."""
    
    print_header("MLOps Auto-Retraining Pipeline Demo")
    print("This demo will showcase the complete pipeline:")
    print("1. Model Training with MLflow")
    print("2. Model Evaluation")
    print("3. API Server Deployment")
    print("4. Streamlit Dashboard")
    print("5. End-to-End Testing")
    
    # Change to project directory
    project_dir = "ai-portfolio/projects/mlops-auto-retrain-gcp"
    if not os.path.exists(project_dir):
        print(f"❌ Project directory {project_dir} not found!")
        return False
    
    os.chdir(project_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Step 1: Train Model
    print_step(1, "Model Training with MLflow")
    if not run_command("uv run python train.py", "Model training"):
        print("❌ Demo failed at training step")
        return False
    
    # Step 2: Evaluate Model
    print_step(2, "Model Evaluation")
    if not run_command("uv run python evaluate.py", "Model evaluation"):
        print("❌ Demo failed at evaluation step")
        return False
    
    # Step 3: Start API Server
    print_step(3, "Starting FastAPI Server")
    print("⚡ Starting API server in background...")
    
    # Start API server in background
    api_process = subprocess.Popen(
        ["uv", "run", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("⏳ Waiting for API server to start...")
    time.sleep(10)
    
    # Test API health
    print("🔍 Testing API health...")
    health_result = test_api_endpoint("http://localhost:8000/health", description="Health check")
    
    if not health_result:
        print("❌ API server failed to start properly")
        api_process.terminate()
        return False
    
    print(f"✅ API Server Status: {health_result.get('status', 'unknown')}")
    print(f"📊 Model Version: {health_result.get('model_version', 'unknown')}")
    
    # Step 4: Test Predictions
    print_step(4, "Testing Predictions")
    
    # Test single prediction
    sample_customer = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": "786.0",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes"
    }
    
    print("🔍 Testing single customer prediction...")
    prediction_result = test_api_endpoint(
        "http://localhost:8000/predict", 
        sample_customer, 
        "Single prediction"
    )
    
    if prediction_result:
        print(f"📊 Churn Probability: {prediction_result['churn_probability']:.1%}")
        print(f"🎯 Prediction: {prediction_result['churn_prediction']}")
        print(f"⚠️ Risk Level: {prediction_result['risk_level']}")
    
    # Test batch prediction
    print("\n🔍 Testing batch prediction...")
    batch_request = {
        "customers": [sample_customer, {
            "SeniorCitizen": 1,
            "tenure": 24,
            "MonthlyCharges": 85.0,
            "TotalCharges": "2040.0",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "Contract": "Two year",
            "PaymentMethod": "Credit card",
            "PaperlessBilling": "No"
        }]
    }
    
    batch_result = test_api_endpoint(
        "http://localhost:8000/batch_predict", 
        batch_request, 
        "Batch prediction"
    )
    
    if batch_result:
        print(f"📊 Total Customers: {batch_result['total_customers']}")
        print(f"⚠️ High Risk Customers: {batch_result['high_risk_count']}")
    
    # Step 5: Start Streamlit Dashboard
    print_step(5, "Starting Streamlit Dashboard")
    print("⚡ Starting Streamlit dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    
    # Start Streamlit in background
    streamlit_process = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(5)
    
    # Step 6: Demo Summary
    print_step(6, "Demo Summary")
    print("🎉 MLOps Pipeline Demo Completed Successfully!")
    print("\n📊 What's Running:")
    print("• FastAPI Server: http://localhost:8000")
    print("• API Documentation: http://localhost:8000/docs")
    print("• Streamlit Dashboard: http://localhost:8501")
    print("• MLflow UI: mlflow ui (run in separate terminal)")
    
    print("\n🔧 Available Endpoints:")
    print("• GET  /health - Health check")
    print("• GET  /model_info - Model information")
    print("• POST /predict - Single prediction")
    print("• POST /batch_predict - Batch predictions")
    
    print("\n📈 Model Performance:")
    if health_result:
        print(f"• Model Version: {health_result.get('model_version', 'N/A')}")
    if prediction_result:
        print(f"• Sample Prediction: {prediction_result['churn_probability']:.1%} churn probability")
    
    print("\n🎯 Next Steps:")
    print("1. Open http://localhost:8501 to use the Streamlit dashboard")
    print("2. Open http://localhost:8000/docs to explore the API")
    print("3. Run 'mlflow ui' to view experiment tracking")
    print("4. Install Docker and GCP SDK for cloud deployment")
    
    print("\n⚠️ To stop the demo:")
    print("Press Ctrl+C to stop this script")
    print("The API and Streamlit servers will continue running")
    
    # Keep the demo running
    try:
        print("\n🔄 Demo is running... Press Ctrl+C to stop")
        while True:
            time.sleep(30)
            # Check if services are still running
            health_check = test_api_endpoint("http://localhost:8000/health")
            if health_check:
                print(f"✅ {datetime.now().strftime('%H:%M:%S')} - Services running normally")
            else:
                print(f"⚠️ {datetime.now().strftime('%H:%M:%S')} - API service may have stopped")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping demo...")
        
        # Clean up processes
        print("🧹 Cleaning up processes...")
        api_process.terminate()
        streamlit_process.terminate()
        
        # Wait for processes to terminate
        api_process.wait(timeout=5)
        streamlit_process.wait(timeout=5)
        
        print("✅ Demo stopped successfully")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
