# Deployment Guide: Production RAG System

This guide outlines the steps to deploy the RAG Production System to cloud environments (AWS/GCP).

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **API Keys**: OpenAI, Cohere, and Arize Phoenix (if using hosted).
- **Domain Name**: For SSL/TLS termination.

---

## ☁️ Option 1: AWS (EC2 + Docker)

### 1. Provision Instance
- Launch an `m5.large` (8GB RAM) instance or higher.
- Open ports: `80 (HTTP)`, `443 (HTTPS)`, `6333 (Qdrant)`, `6006 (Phoenix)`.

### 2. Setup Environment
```bash
sudo apt update && sudo apt install -y docker.io docker-compose
git clone https://github.com/your-repo/rag-production-system.git
cd rag-production-system
cp .env.example .env
# Edit .env with your production keys
```

### 3. Deploy
```bash
cd docker
docker-compose up -d --build
```

---

## ☁️ Option 2: GCP (Cloud Run + Managed Databases)

### 1. Containerize
```bash
gcloud builds submit --tag gcr.io/[PROJECT_ID]/rag-api ..
```

### 2. Deploy Qdrant
- Use **Qdrant Cloud** (Managed) or deploy a persistent GCE instance for Qdrant.

### 3. Deploy API
```bash
gcloud run deploy rag-api \
  --image gcr.io/[PROJECT_ID]/rag-api \
  --set-env-vars="QDRANT_HOST=[QDRANT_IP],OPENAI_API_KEY=[KEY]..." \
  --platform managed
```

---

## 📈 Monitoring & Maintenance

- **Backups**: Schedule nightly snapshots of the `qdrant_data` volume.
- **Scaling**: Use a Load Balancer (ALB/GLB) to scale the `rag-api` service horizontally.
- **Tracing**: Access the Phoenix UI at `http://your-ip:6006` to monitor query quality.
- **RAGAS CI**: Run `npm test` or `pytest evaluation/` in your CI/CD pipeline to prevent regression.
