import os
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from mlflow.exceptions import RestException
from typing import Dict, Any


DATA_PATH = "../../../data/datasets/churn.csv"
df = pd.read_csv(DATA_PATH)


# EDA
print("\n======== EDA ==========")
print("Shape:", df.shape)
print("Null counts:\n", df.isna().sum().sort_values(ascending=False).head())
print("Target balance:\n", df["Churn"].value_counts(normalize=True))

# convert dtypes
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})  # Using replace instead of map
df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")


num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols + ["customerID", "Churn"]]
print("\nNumeric summary:\n", df[num_cols].describe().T[["mean","std","min","max"]])


# Train/val/test split
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].replace({"Yes": 1, "No": 0})  # Using replace instead of map

X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
X_val, X_test,  y_val, y_test  = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)


# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


# Candidate models
models: Dict[str, Any] = {
    "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
    ),
    "grad_boost": GradientBoostingClassifier()
}

mlflow.set_experiment("churn_prediction")

best_run_id, best_auc = None, -np.inf
best_model_name = ""

for name, clf in models.items():
    with mlflow.start_run(run_name=name) as run:
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)

        # ---------- validation metrics ----------
        val_preds = pipe.predict(X_val)
        val_auc: float = float(roc_auc_score(y_val, val_preds))
        val_acc: float = float(accuracy_score(y_val, val_preds))

        # Log metrics as primitive types
        mlflow.log_metrics({
            "val_roc_auc": float(val_auc),
            "val_accuracy": float(val_acc)
        })
        
        mlflow_sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="churn_classifier"
        )

        # track best
        if val_auc > best_auc:
            best_auc = val_auc
            best_run_id = run.info.run_id
            best_model_name = name
        print(f"[{name}]  ROC-AUC={val_auc:.3f}  ACC={val_acc:.3f}")

print(f"\nBest model: {best_model_name} | run: {best_run_id} | ROC-AUC={best_auc:.3f}")

# Get the latest version details
client = MlflowClient()
latest_version = client.search_model_versions(f"name='churn_classifier'")[0]
print(f"Registered model version {latest_version.version} as latest version.")