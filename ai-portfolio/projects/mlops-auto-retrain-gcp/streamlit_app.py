#!/usr/bin/env python3
"""Streamlit web interface for churn prediction."""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"  # Will be updated for production

def main():
    st.set_page_config(
        page_title="Churn Prediction Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Customer Churn Prediction Dashboard")
    st.markdown("**Predict customer churn using our production ML model**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Prediction", "Batch Prediction", "Model Info", "API Health"]
    )
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Model Info":
        model_info_page()
    elif page == "API Health":
        api_health_page()

def single_prediction_page():
    st.header("🔍 Single Customer Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        st.subheader("Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    
    with col2:
        st.subheader("Contract & Billing")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    
    if st.button("🎯 Predict Churn", type="primary"):
        # Prepare data for API
        customer_data = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless_billing
        }
        
        # Make prediction
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=customer_data)
            if response.status_code == 200:
                result = response.json()
                display_prediction_result(result)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def display_prediction_result(result):
    """Display prediction results with visualizations."""
    st.success("✅ Prediction completed!")
    
    # Create columns for results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churn_prob = result["churn_probability"]
        st.metric("Churn Probability", f"{churn_prob:.1%}")
        
    with col2:
        st.metric("Prediction", result["churn_prediction"])
        
    with col3:
        risk_level = result["risk_level"]
        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        st.metric("Risk Level", f"{risk_color.get(risk_level, '⚪')} {risk_level}")
    
    # Probability gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = churn_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.info(f"🤖 Model Version: {result['model_version']} | ⏰ Prediction Time: {result['prediction_timestamp']}")

def batch_prediction_page():
    st.header("📊 Batch Prediction")
    
    st.markdown("Upload a CSV file with customer data for batch predictions.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Convert DataFrame to list of dictionaries
                customers = df.to_dict('records')
                
                # Prepare batch request
                batch_request = {"customers": customers}
                
                # Make batch prediction
                response = requests.post(f"{API_BASE_URL}/batch_predict", json=batch_request)
                
                if response.status_code == 200:
                    results = response.json()
                    display_batch_results(results, df)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def display_batch_results(results, original_df):
    """Display batch prediction results."""
    st.success("✅ Batch prediction completed!")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", results["total_customers"])
    with col2:
        st.metric("High Risk Customers", results["high_risk_count"])
    with col3:
        high_risk_pct = (results["high_risk_count"] / results["total_customers"]) * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    # Convert results to DataFrame
    predictions_df = pd.DataFrame([pred for pred in results["predictions"]])
    
    # Risk distribution chart
    risk_counts = predictions_df["risk_level"].value_counts()
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Risk Level Distribution",
        color_discrete_map={"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📊 Detailed Results")
    display_df = predictions_df[["customer_id", "churn_probability", "churn_prediction", "risk_level"]]
    display_df["churn_probability"] = display_df["churn_probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True)
    
    # Download results
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def model_info_page():
    st.header("🤖 Model Information")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model_info")
        if response.status_code == 200:
            model_info = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Details")
                st.write(f"**Model Name:** {model_info['model_name']}")
                st.write(f"**Version:** {model_info['model_version']}")
                st.write(f"**Type:** {model_info['model_type']}")
            
            with col2:
                st.subheader("Expected Features")
                features = model_info['features_expected']
                for feature in features:
                    st.write(f"• {feature}")
                    
        else:
            st.error(f"Failed to get model info: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")

def api_health_page():
    st.header("🏥 API Health Status")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_info = response.json()
            
            st.success("✅ API is healthy!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Status Information")
                st.write(f"**Status:** {health_info['status']}")
                st.write(f"**Model Loaded:** {health_info['model_loaded']}")
                st.write(f"**Model Version:** {health_info['model_version']}")
            
            with col2:
                st.subheader("Timestamp")
                st.write(f"**Last Check:** {health_info['timestamp']}")
                
        else:
            st.error(f"❌ API Health Check Failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")

if __name__ == "__main__":
    main()
