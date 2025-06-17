# From CSV to Model Registry: A Churn Prediction Pipeline with MLflow and Scikit-learn

**By Aishwarya Jauhari**\
*June 2025*

---

Predicting customer churn is a common and valuable machine learning task. Businesses want to identify customers who are likely to leave so they can intervene in time. However, building a single churn model in a Jupyter notebook isn't enough. What matters is creating a system that's experiment-tracked, repeatable, version-controlled, and ready for deployment.

That's why I built a complete machine learning pipeline for churn prediction using **pandas**, **scikit-learn**, and **MLflow**. In this blog, I will walk through the journey of transforming a simple CSV dataset into a robust, trackable, and registerable machine learning workflow.

We'll explore everything from data cleaning and modeling to logging with MLflow and versioning models in the MLflow Model Registry. This post is intended for readers who are already comfortable with basic machine learning workflows and want to take their projects a step closer to production-level maturity.

---

## Problem Overview: Telco Customer Churn

The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It includes information about a telecommunications company's customers, such as their demographic info, contract details, services they’ve signed up for, and whether they’ve churned or not.

This is a binary classification problem, where the target variable is `Churn`. The dataset is somewhat imbalanced: about 73% of customers stay, and 27% churn. This makes accuracy a poor metric for model evaluation, as we’ll discuss later.

---

## Step 1: Data Cleaning and Preprocessing

The dataset, although clean on the surface, had a few subtle issues:

1. The `TotalCharges` column was incorrectly typed as an object due to blank entries.
2. The `SeniorCitizen` column was encoded as 0/1, while other binary fields were in "Yes"/"No" format.
3. Several categorical fields were not explicitly marked as categorical in the DataFrame.

These needed correction:

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
```

This conversion ensured consistency across the dataset, especially when it comes to encoding and imputation. After identifying numeric and categorical columns, we created a unified preprocessing pipeline using `ColumnTransformer`. This separates the transformations applied to numeric and categorical columns while keeping the final pipeline manageable and modular.

```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
```

This `preprocessor` object becomes the first step in every model pipeline.

---

## Step 2: Training Multiple Models

Instead of relying on a single model, I wanted to compare three classifiers:

- Logistic Regression
- Random Forest
- Gradient Boosting

Each model was wrapped into a `Pipeline` that starts with preprocessing and ends with the classifier.

```python
pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', clf)
])
```

This approach allows us to treat every model run uniformly and also makes it easy to deploy the pipeline later — no need to repeat the same preprocessing logic during inference.

I split the data into training, validation, and test sets. For model comparison, I used **ROC-AUC** as the primary metric. Since the dataset is imbalanced, ROC-AUC gives a more accurate view of the model's ability to distinguish churners from non-churners than simple accuracy.

---

## Step 3: MLflow for Experiment Tracking

Here’s where **MLflow** came into play.

With each model run, I wanted to:

- Record the model type
- Log evaluation metrics (ROC-AUC and accuracy)
- Store the trained pipeline
- Register the best model

MLflow’s `start_run()` method makes it easy to encapsulate each model training session.

```python
with mlflow.start_run(run_name=model_name) as run:
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    auc = roc_auc_score(y_val, preds)

    mlflow.log_metric("val_roc_auc", auc)
    mlflow_sklearn.log_model(pipe, artifact_path="model", registered_model_name="churn_classifier")
```

Each run appears in the MLflow UI, showing metrics, artifacts, parameters, and more. This eliminates the guesswork of “what settings led to which result.”

---

## Why MLflow Matters

MLflow transforms your machine learning workflow from a black box to a transparent, traceable pipeline. It answers key questions like:

- Which model had the best validation ROC-AUC?
- What hyperparameters were used?
- What code and environment was this model trained under?
- Can I reproduce this result?

In a team or production setting, these questions are not optional — they are foundational.

The best part? MLflow makes this process **lightweight**. You don’t need a Kubernetes cluster or an MLOps engineer to start logging experiments and managing models.

---

## Step 4: Model Registry

MLflow doesn’t just stop at experiment tracking. Its **Model Registry** feature lets you register and version models.

In my workflow, the model with the highest validation ROC-AUC was automatically pushed to the registry:

```python
mlflow_sklearn.log_model(..., registered_model_name='churn_classifier')
```

This created a new version of the model each time. Later, I can:

- Assign aliases like `staging`, `production`
- Deploy the latest version without worrying about file paths
- Roll back to previous models easily

To load a model for inference, I just use:

```python
model = mlflow_sklearn.load_model("models:/churn_classifier@staging")
```

This small change makes your code **production-safe** and removes hardcoded run IDs.

---

## Final Thoughts

What started as a churn prediction problem became a lesson in building reproducible ML workflows. With just scikit-learn and MLflow, I was able to:

- Train multiple models in a clean loop
- Track experiments, metrics, and artifacts
- Automatically register and version the best model

MLflow adds almost no overhead to your workflow but gives you the kind of tracking and model management that’s critical for production systems.

If you're not yet using MLflow — start now. It’s the missing piece between academic-style notebooks and reliable ML pipelines.

---



