# Deployment Guide - Vision RAG Agent

This document outlines how to deploy the Vision RAG Agent to Google Cloud Run.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Manual Deployment](#manual-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud` CLI)
- [Docker](https://docs.docker.com/get-docker/) (for local builds)
- Python 3.11+

### GCP Project Setup

1. Create a GCP project or use an existing one:
   ```bash
   gcloud projects create vision-rag-project --name="Vision RAG Agent"
   gcloud config set project vision-rag-project
   ```

2. Enable required APIs:
   ```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     run.googleapis.com \
     containerregistry.googleapis.com \
     secretmanager.googleapis.com
   ```

3. Set up Secret Manager for API keys (recommended):
   ```bash
   # Store OpenRouter API key
   echo -n "your-openrouter-key" | gcloud secrets create openrouter-api-key --data-file=-

   # Store Cohere API key
   echo -n "your-cohere-key" | gcloud secrets create cohere-api-key --data-file=-

   # Grant Cloud Run access to secrets
   gcloud secrets add-iam-policy-binding openrouter-api-key \
     --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"

   gcloud secrets add-iam-policy-binding cohere-api-key \
     --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

---

## Local Development

### Build and Run Locally

```bash
# Build the Docker image
docker build -t vision-rag-agent .

# Run the container
docker run -p 8080:8080 \
  -e OPENROUTER_API_KEY=your-key \
  -e COHERE_API_KEY=your-key \
  vision-rag-agent

# Access at http://localhost:8080
```

### Test the Health Endpoint

```bash
curl http://localhost:8080/_stcore/health
```

---

## Manual Deployment

### Option 1: Using Cloud Build (Recommended)

```bash
# Deploy using cloudbuild.yaml
gcloud builds submit --config=cloudbuild.yaml

# Or with custom substitutions
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=my-vision-rag,_REGION=us-east1,_MEMORY=4Gi
```

### Option 2: Direct Docker Push

```bash
# Set variables
export PROJECT_ID=$(gcloud config get-value project)
export SERVICE_NAME=vision-rag-agent
export REGION=us-central1

# Configure Docker for GCR
gcloud auth configure-docker gcr.io

# Build and push
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME .
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --region $REGION \
  --platform managed \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated \
  --set-env-vars "STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0,STREAMLIT_SERVER_HEADLESS=true"
```

---

## CI/CD Pipeline

### GitHub Actions Setup

The `.github/workflows/deploy.yml` workflow automates deployment on push to `main`.

#### Required GitHub Secrets

Configure these in your repository Settings > Secrets and variables > Actions:

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `WIF_PROVIDER` | Workload Identity Federation provider |
| `WIF_SERVICE_ACCOUNT` | Service account email for WIF |

#### Setting Up Workload Identity Federation (Recommended)

WIF allows GitHub Actions to authenticate without storing service account keys.

```bash
# Create a Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create a provider for GitHub
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Create a service account for deployments
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions Deployer"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Allow GitHub to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_ORG/YOUR_REPO"
```

The `WIF_PROVIDER` secret value will be:
```
projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

The `WIF_SERVICE_ACCOUNT` secret value will be:
```
github-actions-deployer@PROJECT_ID.iam.gserviceaccount.com
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8080` |
| `STREAMLIT_SERVER_PORT` | Streamlit port | `8080` |
| `STREAMLIT_SERVER_ADDRESS` | Bind address | `0.0.0.0` |
| `OPENROUTER_API_KEY` | OpenRouter API key | (required) |
| `COHERE_API_KEY` | Cohere API key | (required) |
| `LANGCHAIN_API_KEY` | LangSmith API key | (optional) |

### Cloud Run Settings

Adjust these in `cloudbuild.yaml` or via `gcloud` flags:

| Setting | Default | Recommendation |
|---------|---------|----------------|
| Memory | 2Gi | Increase for large PDFs |
| CPU | 2 | Increase for faster embedding |
| Min Instances | 0 | Set to 1 to avoid cold starts |
| Max Instances | 10 | Scale based on traffic |
| Concurrency | 80 | Reduce for memory-heavy workloads |
| Timeout | 300s | Increase for long-running queries |

---

## Troubleshooting

### Common Issues

#### 1. Container fails to start

Check logs:
```bash
gcloud run services logs read vision-rag-agent --region us-central1 --limit 50
```

#### 2. Health check fails

The health endpoint is `/_stcore/health`. Ensure:
- Port 8080 is exposed
- Streamlit is configured for headless mode
- Container has sufficient memory

#### 3. API key errors

Verify secrets are mounted correctly:
```bash
gcloud run services describe vision-rag-agent --region us-central1 --format yaml
```

#### 4. Cold start latency

Set minimum instances to 1:
```bash
gcloud run services update vision-rag-agent \
  --region us-central1 \
  --min-instances 1
```

### Viewing Logs

```bash
# Real-time logs
gcloud run services logs tail vision-rag-agent --region us-central1

# Historical logs in Cloud Console
# https://console.cloud.google.com/run/detail/us-central1/vision-rag-agent/logs
```

### Rollback

```bash
# List revisions
gcloud run revisions list --service vision-rag-agent --region us-central1

# Rollback to previous revision
gcloud run services update-traffic vision-rag-agent \
  --region us-central1 \
  --to-revisions PREVIOUS_REVISION=100
```

---

## Architecture

```
GitHub Push (main)
       |
       v
GitHub Actions
  - Lint & Type Check
  - Run Tests
  - Build Docker Image
  - Push to GCR
  - Deploy to Cloud Run
       |
       v
Cloud Run
  - Auto-scaling (0-10 instances)
  - Health checks
  - HTTPS endpoint
       |
       v
Streamlit App (port 8080)
  - Vision RAG Agent
  - ChromaDB (ephemeral)
  - OpenRouter / Cohere APIs
```

---

## Cost Optimization

1. **Use min-instances=0** for development environments
2. **Enable Cloud Run CPU throttling** to reduce idle costs
3. **Use Artifact Registry** instead of Container Registry for lower storage costs
4. **Set appropriate concurrency** to maximize utilization

---

## Security Best Practices

1. Store API keys in Secret Manager, not environment variables
2. Use Workload Identity Federation instead of service account keys
3. Enable VPC connector for private network access if needed
4. Regularly rotate API keys and review IAM permissions
