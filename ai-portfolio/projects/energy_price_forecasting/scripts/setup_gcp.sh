#!/bin/bash

# Setup script for Google Cloud Platform resources
# Run this script to set up the necessary GCP resources for the Energy Price Forecasting project

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
BUCKET_NAME=${GCS_BUCKET_NAME:-"energy-forecasting-data-$(date +%s)"}
SERVICE_ACCOUNT_NAME="energy-forecasting-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🚀 Setting up GCP resources for Energy Price Forecasting"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket Name: $BUCKET_NAME"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set the project
echo "📋 Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Create service account
echo "👤 Creating service account..."
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL &> /dev/null; then
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="Energy Forecasting Service Account" \
        --description="Service account for energy price forecasting project"
else
    echo "Service account already exists"
fi

# Grant necessary roles to service account
echo "🔐 Granting roles to service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.user"

# Create GCS bucket
echo "🪣 Creating GCS bucket..."
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
    
    # Set bucket lifecycle policy
    cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesPrefix": ["raw/"]
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["processed/"]
        }
      }
    ]
  }
}
EOF
    
    gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME
    rm lifecycle.json
    
    echo "✅ Bucket created with lifecycle policy"
else
    echo "Bucket already exists"
fi

# Create bucket folders
echo "📁 Creating bucket folder structure..."
echo "" | gsutil cp - gs://$BUCKET_NAME/raw/.keep
echo "" | gsutil cp - gs://$BUCKET_NAME/processed/.keep
echo "" | gsutil cp - gs://$BUCKET_NAME/models/.keep
echo "" | gsutil cp - gs://$BUCKET_NAME/logs/.keep

# Create service account key (for local development)
echo "🔑 Creating service account key..."
KEY_FILE="gcp-service-account-key.json"
if [ ! -f "$KEY_FILE" ]; then
    gcloud iam service-accounts keys create $KEY_FILE \
        --iam-account=$SERVICE_ACCOUNT_EMAIL
    echo "✅ Service account key created: $KEY_FILE"
    echo "⚠️  Keep this file secure and do not commit it to version control!"
else
    echo "Service account key already exists"
fi

# Create secrets in Secret Manager
echo "🔒 Creating secrets in Secret Manager..."

# Function to create secret if it doesn't exist
create_secret_if_not_exists() {
    local secret_name=$1
    local secret_value=$2
    
    if ! gcloud secrets describe $secret_name &> /dev/null; then
        echo "$secret_value" | gcloud secrets create $secret_name --data-file=-
        echo "✅ Created secret: $secret_name"
    else
        echo "Secret $secret_name already exists"
    fi
}

# Create placeholder secrets (you'll need to update these with real values)
create_secret_if_not_exists "entso-e-api-token" "your-entso-e-token-here"
create_secret_if_not_exists "openweather-api-key" "your-openweather-key-here"
create_secret_if_not_exists "gcs-bucket-name" "$BUCKET_NAME"

# Set up Cloud Build triggers (optional)
echo "🏗️  Setting up Cloud Build..."
if [ -f "../.github/workflows/ci-cd.yml" ]; then
    echo "GitHub Actions workflow found. You can also set up Cloud Build triggers if needed."
fi

# Create environment file
echo "📝 Creating environment configuration..."
cat > .env.gcp << EOF
# GCP Configuration for Energy Price Forecasting
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
GCS_BUCKET_NAME=$BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS=./gcp-service-account-key.json

# API Keys (update with real values)
ENTSO_E_TOKEN=your-entso-e-token-here
OPENWEATHER_API_KEY=your-openweather-key-here

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_LOCATION=gs://$BUCKET_NAME/mlruns
EOF

echo "✅ Environment file created: .env.gcp"

# Display next steps
echo ""
echo "🎉 GCP setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update the API keys in Secret Manager:"
echo "   - ENTSO-E API token: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html"
echo "   - OpenWeatherMap API key: https://openweathermap.org/api"
echo ""
echo "2. Update secrets with real values:"
echo "   gcloud secrets versions add entso-e-api-token --data-file=<(echo 'your-real-token')"
echo "   gcloud secrets versions add openweather-api-key --data-file=<(echo 'your-real-key')"
echo ""
echo "3. Set up GitHub repository secrets for CI/CD:"
echo "   - GCP_PROJECT_ID: $PROJECT_ID"
echo "   - GCP_SA_KEY: $(cat $KEY_FILE | base64 -w 0)"
echo "   - GCS_BUCKET_NAME: $BUCKET_NAME"
echo "   - ENTSO_E_TOKEN: your-entso-e-token"
echo "   - OPENWEATHER_API_KEY: your-openweather-key"
echo ""
echo "4. Test the setup:"
echo "   export GOOGLE_APPLICATION_CREDENTIALS=./gcp-service-account-key.json"
echo "   python -c \"from google.cloud import storage; print('GCS connection successful!')\""
echo ""
echo "🔗 Useful links:"
echo "   - GCS Bucket: https://console.cloud.google.com/storage/browser/$BUCKET_NAME"
echo "   - Cloud Run: https://console.cloud.google.com/run?project=$PROJECT_ID"
echo "   - Secret Manager: https://console.cloud.google.com/security/secret-manager?project=$PROJECT_ID"
