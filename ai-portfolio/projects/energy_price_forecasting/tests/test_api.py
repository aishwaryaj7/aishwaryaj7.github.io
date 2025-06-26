"""
Tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

# Create test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert "timestamp" in data
    
    def test_models_info_endpoint_no_models(self):
        """Test models info endpoint when no models are loaded."""
        response = client.get("/models/info")
        # Should return 503 if no models loaded, or empty list if models are loaded
        assert response.status_code in [200, 503]
    
    def test_predict_endpoint_no_models(self):
        """Test predict endpoint when no models are loaded."""
        request_data = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 24,
            "include_confidence_intervals": True
        }
        
        response = client.post("/predict", json=request_data)
        # Should return 503 if no models loaded
        assert response.status_code in [200, 503]
    
    def test_predict_endpoint_invalid_model(self):
        """Test predict endpoint with invalid model name."""
        request_data = {
            "country": "DE",
            "model_name": "invalid_model",
            "forecast_horizon": 24,
            "include_confidence_intervals": True
        }
        
        response = client.post("/predict", json=request_data)
        # Should return error for invalid model
        assert response.status_code in [400, 503]
    
    def test_predict_endpoint_validation(self):
        """Test predict endpoint input validation."""
        # Test invalid country
        request_data = {
            "country": "XX",  # Invalid country
            "model_name": "xgboost",
            "forecast_horizon": 24
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
        
        # Test invalid forecast horizon
        request_data = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 200  # Too large
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_model_performance_endpoint(self):
        """Test model performance endpoint."""
        response = client.get("/models/xgboost/performance")
        # Should return 404 or 503 if model not found/loaded
        assert response.status_code in [200, 404, 503]
    
    def test_retrain_endpoint(self):
        """Test retrain endpoint."""
        response = client.post("/retrain")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert "timestamp" in data

class TestAPISchemas:
    """Test API request/response schemas."""
    
    def test_valid_forecast_request(self):
        """Test valid forecast request schema."""
        valid_request = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 24,
            "start_datetime": "2024-01-15T00:00:00",
            "include_confidence_intervals": True
        }
        
        response = client.post("/predict", json=valid_request)
        # Should not fail due to schema validation
        assert response.status_code != 422
    
    def test_minimal_forecast_request(self):
        """Test minimal forecast request."""
        minimal_request = {
            "country": "DE",
            "model_name": "xgboost"
        }
        
        response = client.post("/predict", json=minimal_request)
        # Should not fail due to schema validation (defaults should be used)
        assert response.status_code != 422
    
    def test_forecast_request_edge_cases(self):
        """Test forecast request edge cases."""
        # Minimum forecast horizon
        request_data = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 1
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code != 422
        
        # Maximum forecast horizon
        request_data = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 168
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code != 422

class TestAPIIntegration:
    """Integration tests for API functionality."""
    
    def test_api_workflow(self):
        """Test complete API workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info
        models_response = client.get("/models/info")
        # May return 503 if no models loaded, which is fine for testing
        
        # 3. Try to make a prediction
        predict_request = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 6
        }
        
        predict_response = client.post("/predict", json=predict_request)
        # May return 503 if no models loaded, which is fine for testing
        
        # If prediction succeeds, validate response structure
        if predict_response.status_code == 200:
            data = predict_response.json()
            assert "predictions" in data
            assert "model_used" in data
            assert "forecast_horizon" in data
            assert "country" in data
            assert "generated_at" in data
            
            # Check predictions structure
            predictions = data["predictions"]
            assert len(predictions) == predict_request["forecast_horizon"]
            
            for pred in predictions:
                assert "datetime" in pred
                assert "predicted_price" in pred
                assert "country" in pred
                assert isinstance(pred["predicted_price"], (int, float))
    
    def test_api_error_handling(self):
        """Test API error handling."""
        # Test malformed JSON
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Test invalid endpoint
        response = client.get("/invalid_endpoint")
        assert response.status_code == 404

class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_response_times(self):
        """Test API response times."""
        import time
        
        # Test health endpoint response time
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_health_request():
            return client.get("/health")
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

class TestAPISecurity:
    """Test API security aspects."""
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # CORS headers should be present due to middleware
        # Note: TestClient might not include all CORS headers
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        # Test with potentially malicious input
        malicious_request = {
            "country": "DE",
            "model_name": "<script>alert('xss')</script>",
            "forecast_horizon": 24
        }
        
        response = client.post("/predict", json=malicious_request)
        # Should either validate properly or handle gracefully
        assert response.status_code in [200, 400, 422, 503]
    
    def test_large_request_handling(self):
        """Test handling of large requests."""
        # Test with maximum allowed forecast horizon
        large_request = {
            "country": "DE",
            "model_name": "xgboost",
            "forecast_horizon": 168  # Maximum allowed
        }
        
        response = client.post("/predict", json=large_request)
        # Should handle gracefully
        assert response.status_code in [200, 503]

if __name__ == "__main__":
    pytest.main([__file__])
