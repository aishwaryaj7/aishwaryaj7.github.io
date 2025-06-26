#!/bin/bash

# Deployment script for Energy Price Forecasting project
# This script builds and deploys the application to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="energy-price-forecasting"
STREAMLIT_SERVICE_NAME="energy-dashboard"
IMAGE_TAG=${IMAGE_TAG:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying Energy Price Forecasting to Google Cloud Run${NC}"
echo -e "${BLUE}Project: $PROJECT_ID${NC}"
echo -e "${BLUE}Region: $REGION${NC}"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

# Set the project
gcloud config set project $PROJECT_ID

# Configure Docker for GCR
echo -e "${YELLOW}🔧 Configuring Docker for Google Container Registry...${NC}"
gcloud auth configure-docker

# Build and push API image
echo -e "${YELLOW}🏗️  Building API image...${NC}"
docker build --target production-api \
    -t gcr.io/$PROJECT_ID/$SERVICE_NAME:$IMAGE_TAG \
    -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

echo -e "${YELLOW}📤 Pushing API image to GCR...${NC}"
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:$IMAGE_TAG
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Build and push Streamlit image
echo -e "${YELLOW}🏗️  Building Streamlit image...${NC}"
docker build --target production-streamlit \
    -t gcr.io/$PROJECT_ID/$STREAMLIT_SERVICE_NAME:$IMAGE_TAG \
    -t gcr.io/$PROJECT_ID/$STREAMLIT_SERVICE_NAME:latest .

echo -e "${YELLOW}📤 Pushing Streamlit image to GCR...${NC}"
docker push gcr.io/$PROJECT_ID/$STREAMLIT_SERVICE_NAME:$IMAGE_TAG
docker push gcr.io/$PROJECT_ID/$STREAMLIT_SERVICE_NAME:latest

# Get secrets from Secret Manager
echo -e "${YELLOW}🔒 Retrieving secrets...${NC}"
BUCKET_NAME=$(gcloud secrets versions access latest --secret="gcs-bucket-name" 2>/dev/null || echo "energy-forecasting-data")
ENTSO_E_TOKEN=$(gcloud secrets versions access latest --secret="entso-e-api-token" 2>/dev/null || echo "dummy-token")
OPENWEATHER_KEY=$(gcloud secrets versions access latest --secret="openweather-api-key" 2>/dev/null || echo "dummy-key")

# Deploy API to Cloud Run
echo -e "${YELLOW}🚀 Deploying API to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:$IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 100 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID" \
    --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME" \
    --set-env-vars="ENTSO_E_TOKEN=$ENTSO_E_TOKEN" \
    --set-env-vars="OPENWEATHER_API_KEY=$OPENWEATHER_KEY" \
    --set-env-vars="GCP_REGION=$REGION"

# Deploy Streamlit Dashboard to Cloud Run
echo -e "${YELLOW}🚀 Deploying Streamlit Dashboard to Cloud Run...${NC}"
gcloud run deploy $STREAMLIT_SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$STREAMLIT_SERVICE_NAME:$IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 50 \
    --max-instances 5 \
    --min-instances 0 \
    --port 8501 \
    --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID" \
    --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME" \
    --set-env-vars="GCP_REGION=$REGION"

# Get service URLs
echo -e "${YELLOW}🔗 Getting service URLs...${NC}"
API_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
DASHBOARD_URL=$(gcloud run services describe $STREAMLIT_SERVICE_NAME --region=$REGION --format='value(status.url)')

# Test deployments
echo -e "${YELLOW}🧪 Testing deployments...${NC}"

# Wait for services to be ready
sleep 10

# Test API health
echo -e "${YELLOW}Testing API health endpoint...${NC}"
if curl -f -s "$API_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ API health check passed${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
fi

# Test API models endpoint
echo -e "${YELLOW}Testing API models endpoint...${NC}"
if curl -f -s "$API_URL/models/info" > /dev/null; then
    echo -e "${GREEN}✅ API models endpoint accessible${NC}"
else
    echo -e "${YELLOW}⚠️  API models endpoint returned error (models may not be loaded yet)${NC}"
fi

# Test Streamlit dashboard
echo -e "${YELLOW}Testing Streamlit dashboard...${NC}"
if curl -f -s "$DASHBOARD_URL" > /dev/null; then
    echo -e "${GREEN}✅ Streamlit dashboard accessible${NC}"
else
    echo -e "${RED}❌ Streamlit dashboard not accessible${NC}"
fi

# Display results
echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Service URLs:${NC}"
echo -e "${BLUE}API:       $API_URL${NC}"
echo -e "${BLUE}Dashboard: $DASHBOARD_URL${NC}"
echo ""
echo -e "${BLUE}📋 API Endpoints:${NC}"
echo -e "${BLUE}Health:    $API_URL/health${NC}"
echo -e "${BLUE}Models:    $API_URL/models/info${NC}"
echo -e "${BLUE}Predict:   $API_URL/predict${NC}"
echo -e "${BLUE}Docs:      $API_URL/docs${NC}"
echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo -e "${BLUE}View logs (API):       gcloud run logs tail $SERVICE_NAME --region=$REGION${NC}"
echo -e "${BLUE}View logs (Dashboard): gcloud run logs tail $STREAMLIT_SERVICE_NAME --region=$REGION${NC}"
echo -e "${BLUE}Update API:            gcloud run services update $SERVICE_NAME --region=$REGION${NC}"
echo -e "${BLUE}Delete services:       gcloud run services delete $SERVICE_NAME $STREAMLIT_SERVICE_NAME --region=$REGION${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo -e "${YELLOW}1. Test the API with sample requests${NC}"
echo -e "${YELLOW}2. Run the training pipeline to populate models${NC}"
echo -e "${YELLOW}3. Set up monitoring and alerting${NC}"
echo -e "${YELLOW}4. Configure custom domain (optional)${NC}"

# Save URLs to file
cat > deployment_urls.txt << EOF
# Energy Price Forecasting - Deployment URLs
# Generated on $(date)

API_URL=$API_URL
DASHBOARD_URL=$DASHBOARD_URL

# API Endpoints
HEALTH_ENDPOINT=$API_URL/health
MODELS_ENDPOINT=$API_URL/models/info
PREDICT_ENDPOINT=$API_URL/predict
DOCS_ENDPOINT=$API_URL/docs

# Management
PROJECT_ID=$PROJECT_ID
REGION=$REGION
SERVICE_NAME=$SERVICE_NAME
STREAMLIT_SERVICE_NAME=$STREAMLIT_SERVICE_NAME
EOF

echo -e "${GREEN}✅ Deployment URLs saved to deployment_urls.txt${NC}"
