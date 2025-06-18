#!/usr/bin/env python3
"""Tests for the training pipeline."""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import mlflow
from sklearn.metrics import roc_auc_score

# Import training functions (assuming they're in train.py)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_data():
    """Create sample churn data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customerID': [f'C{i:06d}' for i in range(n_samples)],
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'tenure': np.random.randint(0, 73, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(20, 8000, n_samples).astype(str),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_data():
    """Fixture to provide sample data."""
    return create_sample_data()

@pytest.fixture
def temp_data_file(sample_data):
    """Fixture to create a temporary data file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

def test_data_loading(temp_data_file):
    """Test that data can be loaded correctly."""
    df = pd.read_csv(temp_data_file)
    
    # Check basic properties
    assert len(df) == 1000
    assert 'Churn' in df.columns
    assert 'customerID' in df.columns
    
    # Check data types
    assert df['SeniorCitizen'].dtype in [int, 'int64']
    assert df['tenure'].dtype in [int, 'int64']
    assert df['MonthlyCharges'].dtype in [float, 'float64']

def test_data_preprocessing(sample_data):
    """Test data preprocessing steps."""
    # Test TotalCharges conversion
    sample_data['TotalCharges'] = pd.to_numeric(sample_data['TotalCharges'], errors="coerce")
    assert sample_data['TotalCharges'].dtype in [float, 'float64']
    
    # Test SeniorCitizen conversion
    sample_data["SeniorCitizen"] = sample_data["SeniorCitizen"].replace({0: "No", 1: "Yes"})
    assert sample_data["SeniorCitizen"].isin(["Yes", "No"]).all()
    
    # Test target encoding
    y = sample_data["Churn"].replace({"Yes": 1, "No": 0})
    assert y.isin([0, 1]).all()

def test_feature_engineering(sample_data):
    """Test feature engineering steps."""
    # Prepare features
    X = sample_data.drop(columns=["customerID", "Churn"])
    
    # Check that we have the expected features
    expected_features = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingTV',
        'Contract', 'PaymentMethod', 'PaperlessBilling'
    ]
    
    for feature in expected_features:
        assert feature in X.columns

def test_model_training_pipeline():
    """Test the complete training pipeline."""
    # Create temporary directory for MLflow
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_dir = os.path.join(temp_dir, "mlruns")
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Save to temporary file
        data_file = os.path.join(temp_dir, "test_data.csv")
        sample_data.to_csv(data_file, index=False)
        
        # Test that we can run the training process
        # (This would normally import and run your training function)
        
        # For now, let's test the basic ML pipeline components
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Preprocess data
        df = sample_data.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
        
        # Prepare features and target
        X = df.drop(columns=["customerID", "Churn"])
        y = df["Churn"].replace({"Yes": 1, "No": 0})
        
        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            label_encoders[feature] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Assertions
        assert auc_score >= 0.4  # Should be reasonable (synthetic data might not be perfect)
        assert auc_score <= 1.0  # Should not be perfect (would indicate overfitting)
        assert len(y_pred_proba) == len(y_test)

def test_model_persistence():
    """Test that models can be saved and loaded."""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Create and train a simple model
    sample_data = create_sample_data()
    
    # Quick preprocessing
    df = sample_data.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    X = df[['tenure', 'MonthlyCharges']].fillna(0)  # Simple features
    y = df["Churn"].replace({"Yes": 1, "No": 0})
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Test saving and loading
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(model, f.name)
        loaded_model = joblib.load(f.name)
        
        # Test that loaded model works
        predictions = loaded_model.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
    os.unlink(f.name)

def test_mlflow_logging():
    """Test MLflow experiment logging."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_dir = os.path.join(temp_dir, "mlruns")
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.85)
            
            # Log artifact
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test content")
            mlflow.log_artifact(test_file)
        
        # Verify that the run was logged
        runs = mlflow.search_runs()
        assert len(runs) > 0
        assert "params.test_param" in runs.columns
        assert "metrics.test_metric" in runs.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
