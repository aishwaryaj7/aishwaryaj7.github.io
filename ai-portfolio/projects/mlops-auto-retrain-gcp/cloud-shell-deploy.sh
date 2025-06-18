#!/bin/bash

# Cloud Shell Deployment Script
# Run this in Google Cloud Shell for easy deployment

set -e

echo "🚀 MLOps Deployment via Google Cloud Shell"
echo "============================================"

# Check if we're in Cloud Shell
if [ -z "$CLOUD_SHELL" ]; then
    echo "⚠️ This script is designed for Google Cloud Shell"
    echo "🌐 Open: https://shell.cloud.google.com"
    echo "📋 Then run this script"
fi

# Get project ID
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "❌ Please set your project ID:"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

PROJECT_ID=$GOOGLE_CLOUD_PROJECT
SERVICE_NAME="churn-prediction-api"
REGION="us-central1"

echo "📋 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"

# Enable APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet

# Clone repository if not already present
if [ ! -d "aishwaryaj7.github.io" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/aishwaryaj7/aishwaryaj7.github.io.git
fi

cd aishwaryaj7.github.io/ai-portfolio/projects/mlops-auto-retrain-gcp

# Build using Cloud Build
echo "🏗️ Building with Cloud Build..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --set-env-vars="ENVIRONMENT=production"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format='value(status.url)')

echo "✅ Deployment completed!"
echo "🌐 Service URL: $SERVICE_URL"

# Test the deployment
echo "🧪 Testing deployment..."
curl -f "$SERVICE_URL/health"

echo ""
echo "🎉 Success! Your MLOps API is live at:"
echo "   $SERVICE_URL"
echo ""
echo "📚 Try these endpoints:"
echo "   $SERVICE_URL/docs - API documentation"
echo "   $SERVICE_URL/health - Health check"
