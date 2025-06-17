#!/usr/bin/env python3
"""Model evaluation script for churn prediction."""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_PATH = "data/churn.csv"
MODEL_NAME = "churn_classifier"

def load_latest_model():
    """Load the latest registered model from MLflow."""
    client = MlflowClient()

    # Get the latest version
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if not latest_versions:
        raise ValueError(f"No model versions found for {MODEL_NAME}")

    latest_version = latest_versions[0]
    model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"

    print(f"Loading model: {MODEL_NAME} version {latest_version.version}")
    model = mlflow.sklearn.load_model(model_uri)

    return model, latest_version

def evaluate_model():
    """Evaluate the latest model on test data."""
    print("🔍 EVALUATING CHURN PREDICTION MODEL")
    print("=" * 50)

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Preprocess (same as training)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

    # Split data (same random state as training)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].replace({"Yes": 1, "No": 0})

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    # Load model
    model, model_version = load_latest_model()

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Print results
    print(f"\n📊 MODEL PERFORMANCE METRICS")
    print(f"Model Version: {model_version.version}")
    print(f"Test Set Size: {len(y_test)} samples")
    print("-" * 30)

    for metric, value in metrics.items():
        print(f"{metric.upper():<12}: {value:.4f}")

    # Classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 CONFUSION MATRIX")
    print("-" * 20)
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")

    # Business metrics
    total_customers = len(y_test)
    actual_churners = sum(y_test)
    predicted_churners = sum(y_pred)
    correctly_identified = sum((y_test == 1) & (y_pred == 1))

    print(f"\n💼 BUSINESS IMPACT ANALYSIS")
    print("-" * 30)
    print(f"Total customers in test:     {total_customers}")
    print(f"Actual churners:             {actual_churners} ({actual_churners/total_customers:.1%})")
    print(f"Predicted churners:          {predicted_churners} ({predicted_churners/total_customers:.1%})")
    print(f"Correctly identified:        {correctly_identified} ({correctly_identified/actual_churners:.1%} of actual)")

    # Model quality assessment
    print(f"\n🎯 MODEL QUALITY ASSESSMENT")
    print("-" * 30)

    if metrics['roc_auc'] >= 0.8:
        quality = "EXCELLENT"
    elif metrics['roc_auc'] >= 0.7:
        quality = "GOOD"
    elif metrics['roc_auc'] >= 0.6:
        quality = "FAIR"
    else:
        quality = "POOR"

    print(f"Overall Quality: {quality}")
    print(f"ROC-AUC Score: {metrics['roc_auc']:.3f}")

    if metrics['roc_auc'] >= 0.7:
        print("✅ Model meets production quality standards")
    else:
        print("❌ Model needs improvement before production deployment")

    # Log evaluation metrics to MLflow
    with mlflow.start_run(run_name=f"evaluation_v{model_version.version}"):
        mlflow.log_metrics({
            f"test_{k}": v for k, v in metrics.items()
        })
        mlflow.log_param("model_version", model_version.version)
        mlflow.log_param("test_set_size", len(y_test))

        print(f"\n📝 Evaluation metrics logged to MLflow")

    return metrics, model_version

if __name__ == "__main__":
    try:
        metrics, version = evaluate_model()
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Model version {version.version} achieved ROC-AUC: {metrics['roc_auc']:.3f}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        exit(1)