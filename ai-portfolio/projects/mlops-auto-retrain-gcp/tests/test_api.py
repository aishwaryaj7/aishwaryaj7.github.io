#!/usr/bin/env python3
"""Tests for the FastAPI application."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from serve import app, load_model

# Load model before running tests
try:
    load_model()
    print("Model loaded for testing")
except Exception as e:
    print(f"Warning: Could not load model for testing: {e}")

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "timestamp" in data

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model_version" in data
    assert "status" in data

def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "features_expected" in data

def test_single_prediction():
    """Test single customer prediction."""
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
    
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "model_version" in data
    assert "prediction_timestamp" in data
    
    # Validate data types and ranges
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in ["Yes", "No"]
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_batch_prediction():
    """Test batch prediction."""
    sample_customers = [
        {
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
        },
        {
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
        }
    ]
    
    batch_request = {"customers": sample_customers}
    response = client.post("/batch_predict", json=batch_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "total_customers" in data
    assert "high_risk_count" in data
    assert "model_version" in data
    
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert 0 <= data["high_risk_count"] <= 2

def test_invalid_prediction_data():
    """Test prediction with invalid data."""
    invalid_customer = {
        "SeniorCitizen": "invalid",  # Should be 0 or 1
        "tenure": -5,  # Should be positive
        "MonthlyCharges": "not_a_number"  # Should be numeric
    }
    
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422  # Validation error

def test_missing_required_fields():
    """Test prediction with missing required fields."""
    incomplete_customer = {
        "SeniorCitizen": 0,
        "tenure": 12
        # Missing other required fields
    }
    
    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422  # Validation error

def test_prediction_consistency():
    """Test that predictions are consistent for the same input."""
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
    
    # Make multiple predictions with the same data
    response1 = client.post("/predict", json=sample_customer)
    response2 = client.post("/predict", json=sample_customer)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    # Predictions should be identical
    assert data1["churn_probability"] == data2["churn_probability"]
    assert data1["churn_prediction"] == data2["churn_prediction"]
    assert data1["risk_level"] == data2["risk_level"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
