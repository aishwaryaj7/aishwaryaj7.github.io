#!/bin/bash

# Quick GCP Deployment Script
# This script deploys the MLOps pipeline to Google Cloud Run

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 MLOps Quick Deployment to GCP${NC}"
echo "=================================================="

# Check if project ID is set
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${RED}❌ Please set GCP_PROJECT_ID environment variable${NC}"
    echo "Example: export GCP_PROJECT_ID='your-project-id'"
    exit 1
fi

echo -e "${YELLOW}📋 Project ID: $GCP_PROJECT_ID${NC}"

# Configuration
SERVICE_NAME="churn-prediction-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME"

echo -e "${YELLOW}🔧 Checking prerequisites...${NC}"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}🔐 Please authenticate with GCP:${NC}"
    gcloud auth login
fi

# Set project
gcloud config set project $GCP_PROJECT_ID

echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet

echo -e "${YELLOW}🐳 Configuring Docker...${NC}"
gcloud auth configure-docker --quiet

echo -e "${YELLOW}🏗️ Building and deploying...${NC}"

# Build and submit to Cloud Build
gcloud builds submit --tag $IMAGE_NAME

echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
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

echo -e "${GREEN}✅ Deployment completed!${NC}"
echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"

echo -e "${YELLOW}🧪 Testing deployment...${NC}"

# Test health endpoint
if curl -f "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    exit 1
fi

# Test prediction endpoint
echo -e "${YELLOW}🔍 Testing prediction...${NC}"
SAMPLE_DATA='{
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
}'

PREDICTION_RESULT=$(curl -s -X POST "$SERVICE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$SAMPLE_DATA")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Prediction test passed${NC}"
    echo "Sample prediction result:"
    echo "$PREDICTION_RESULT" | python3 -m json.tool
else
    echo -e "${RED}❌ Prediction test failed${NC}"
fi

echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo ""
echo "📊 Your MLOps API is now live at:"
echo "   $SERVICE_URL"
echo ""
echo "🔧 Available endpoints:"
echo "   GET  $SERVICE_URL/health"
echo "   GET  $SERVICE_URL/docs"
echo "   POST $SERVICE_URL/predict"
echo "   POST $SERVICE_URL/batch_predict"
echo ""
echo "💡 Next steps:"
echo "   1. Visit $SERVICE_URL/docs for API documentation"
echo "   2. Test predictions using the interactive docs"
echo "   3. Set up monitoring and alerting"
echo "   4. Configure custom domain (optional)"
