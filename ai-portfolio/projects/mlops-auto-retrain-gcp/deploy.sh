#!/bin/bash

# MLOps Deployment Script for GCP
# This script deploys the churn prediction service to Google Cloud Run

set -e  # Exit on any error

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME="churn-prediction-api"
REGION="us-central1"
REGISTRY="gcr.io"
IMAGE_NAME="$REGISTRY/$PROJECT_ID/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting MLOps Deployment Pipeline${NC}"
echo "=================================================="

# Check if required tools are installed
check_dependencies() {
    echo -e "${YELLOW}📋 Checking dependencies...${NC}"
    
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not found. Please install Google Cloud SDK.${NC}"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All dependencies found${NC}"
}

# Authenticate with GCP
authenticate_gcp() {
    echo -e "${YELLOW}🔐 Authenticating with GCP...${NC}"
    
    # Check if already authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        echo -e "${YELLOW}Please authenticate with GCP:${NC}"
        gcloud auth login
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    echo -e "${GREEN}✅ Authenticated with project: $PROJECT_ID${NC}"
}

# Enable required APIs
enable_apis() {
    echo -e "${YELLOW}🔧 Enabling required GCP APIs...${NC}"
    
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    
    echo -e "${GREEN}✅ APIs enabled${NC}"
}

# Configure Docker for GCR
configure_docker() {
    echo -e "${YELLOW}🐳 Configuring Docker for Google Container Registry...${NC}"
    
    gcloud auth configure-docker
    
    echo -e "${GREEN}✅ Docker configured${NC}"
}

# Build Docker image
build_image() {
    echo -e "${YELLOW}🏗️ Building Docker image...${NC}"
    
    # Get git commit hash for tagging
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    
    # Build image with multiple tags
    docker build -t $IMAGE_NAME:$GIT_COMMIT .
    docker build -t $IMAGE_NAME:latest .
    
    echo -e "${GREEN}✅ Image built: $IMAGE_NAME:$GIT_COMMIT${NC}"
}

# Push image to registry
push_image() {
    echo -e "${YELLOW}📤 Pushing image to Google Container Registry...${NC}"
    
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    
    docker push $IMAGE_NAME:$GIT_COMMIT
    docker push $IMAGE_NAME:latest
    
    echo -e "${GREEN}✅ Image pushed to registry${NC}"
}

# Deploy to Cloud Run
deploy_service() {
    echo -e "${YELLOW}🚀 Deploying to Google Cloud Run...${NC}"
    
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME:$GIT_COMMIT \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 1 \
        --max-instances 10 \
        --set-env-vars="ENVIRONMENT=production" \
        --set-env-vars="MODEL_VERSION=latest"
    
    echo -e "${GREEN}✅ Service deployed${NC}"
}

# Get service URL and test
test_deployment() {
    echo -e "${YELLOW}🧪 Testing deployment...${NC}"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$REGION \
        --format='value(status.url)')
    
    echo -e "${GREEN}📍 Service URL: $SERVICE_URL${NC}"
    
    # Test health endpoint
    echo "Testing health endpoint..."
    if curl -f "$SERVICE_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${RED}❌ Health check failed${NC}"
        exit 1
    fi
    
    # Test prediction endpoint
    echo "Testing prediction endpoint..."
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
    
    if curl -f -X POST "$SERVICE_URL/predict" \
        -H "Content-Type: application/json" \
        -d "$SAMPLE_DATA" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Prediction endpoint working${NC}"
    else
        echo -e "${RED}❌ Prediction endpoint failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🎉 Deployment successful!${NC}"
    echo -e "${GREEN}Service is available at: $SERVICE_URL${NC}"
}

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up local Docker images...${NC}"
    
    # Remove local images to save space
    docker rmi $IMAGE_NAME:latest 2>/dev/null || true
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    docker rmi $IMAGE_NAME:$GIT_COMMIT 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Main deployment pipeline
main() {
    echo -e "${GREEN}Starting deployment for project: $PROJECT_ID${NC}"
    
    check_dependencies
    authenticate_gcp
    enable_apis
    configure_docker
    build_image
    push_image
    deploy_service
    test_deployment
    cleanup
    
    echo -e "${GREEN}🎉 Deployment pipeline completed successfully!${NC}"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        check_dependencies
        build_image
        ;;
    "push")
        check_dependencies
        configure_docker
        push_image
        ;;
    "test")
        test_deployment
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {deploy|build|push|test|cleanup}"
        echo "  deploy  - Full deployment pipeline (default)"
        echo "  build   - Build Docker image only"
        echo "  push    - Push image to registry only"
        echo "  test    - Test deployed service only"
        echo "  cleanup - Clean up local Docker images"
        exit 1
        ;;
esac
