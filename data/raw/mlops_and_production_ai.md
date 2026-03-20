# MLOps and Production AI Systems

## What is MLOps?

MLOps (Machine Learning Operations) is a set of practices that combines ML, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

The core problem MLOps solves: most ML projects fail not because the model doesn't work, but because getting a model from a Jupyter notebook to a reliable, scalable, maintainable production system is extremely hard.

**Key MLOps goals:**
- Reproducibility: every training run should be reproducible
- Automation: CI/CD for ML pipelines
- Monitoring: catch data drift, model degradation, and failures early
- Versioning: track data, code, and model versions together
- Collaboration: streamline handoff between data scientists and engineers

---

## The ML System Lifecycle

### 1. Data Management
**Data versioning:** DVC (Data Version Control) tracks datasets like Git tracks code.
**Data validation:** Great Expectations, TFX Data Validation check schema and statistics.
**Feature stores:** Feast, Tecton centralize feature computation and prevent training-serving skew.

### 2. Experiment Tracking
Track every training run with:
- Hyperparameters
- Metrics (loss, accuracy, etc.) at each epoch
- Code version (git hash)
- Dataset version
- Artifacts (model checkpoints, plots)

**Tools:** MLflow, Weights & Biases (wandb), Neptune, Comet ML

**Example MLflow run:**
```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("val_accuracy", 0.92)
    mlflow.sklearn.log_model(model, "model")
```

### 3. Model Registry
Central repository for model versions:
- Staging → Production → Archived lifecycle
- Metadata: who trained it, on what data, with what metrics
- Tools: MLflow Model Registry, Vertex AI Model Registry

### 4. Serving Infrastructure

**REST API serving:**
- FastAPI / Flask: lightweight Python servers
- TorchServe: PyTorch-native serving
- TF Serving: TensorFlow-native serving
- BentoML: framework-agnostic ML serving

**Batch inference:**
- Run predictions on large datasets offline
- Apache Spark ML, SageMaker Batch Transform

**Streaming inference:**
- Kafka + Flink for real-time feature computation and prediction

**Serverless:**
- AWS Lambda, Google Cloud Functions for low-traffic endpoints
- No server management, scales to zero

### 5. Containerization and Orchestration

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes:**
- Manages containers across a cluster
- Horizontal Pod Autoscaler (HPA): scale based on CPU/memory or custom metrics
- Helm charts: package Kubernetes manifests

**Cloud ML platforms:**
- AWS SageMaker: end-to-end ML platform
- Google Vertex AI: training, serving, monitoring
- Azure ML: similar full-stack offering

---

## CI/CD for ML (MLOps Pipelines)

Classical CI/CD: code changes → build → test → deploy
ML CI/CD also includes: data changes → retrain → evaluate → deploy (if better)

### Kubeflow Pipelines
Define ML pipelines as Python functions decorated with `@component`. Runs on Kubernetes.

### Apache Airflow
DAG-based workflow orchestration. Widely used for data pipelines that include ML steps.

### GitHub Actions for ML
- Trigger training on data changes (DVC + GitHub Actions)
- Run model evaluation and post results as PR comments
- Auto-deploy if new model beats baseline

---

## Monitoring ML Systems

### Types of Drift

**Data Drift (Covariate Shift):**
The distribution of input features changes over time.
- Example: model trained on summer traffic data deployed in winter
- Detect with: KS test, PSI (Population Stability Index), Wasserstein distance
- Tools: Evidently AI, WhyLogs, NannyML

**Concept Drift:**
The relationship between inputs and outputs changes.
- Example: "affordable rent" meant $800/month in 2015, means $1200 in 2023
- Harder to detect than data drift — requires ground truth labels

**Label Drift:**
The distribution of output labels changes.

**Model Performance Degradation:**
The most important signal — track accuracy, F1, AUC on production data.

### What to Monitor
- Request volume and latency (P50, P95, P99)
- Prediction distribution (watch for sudden shifts)
- Feature statistics (mean, std, null rates)
- Model confidence / uncertainty
- Business metrics (downstream impact on revenue, clicks, etc.)

### Observability Stack
- **Metrics:** Prometheus + Grafana
- **Logs:** ELK stack (Elasticsearch + Logstash + Kibana) or Datadog
- **Traces:** OpenTelemetry, Jaeger
- **Alerts:** PagerDuty, OpsGenie

---

## Feature Stores

A feature store is a central repository for ML features:
- **Offline store**: historical features for training (usually a data warehouse or lake)
- **Online store**: low-latency features for inference (usually Redis or DynamoDB)

**Benefits:**
- No training-serving skew (same code computes features in both)
- Feature reuse across teams
- Point-in-time correctness for historical queries

**Popular feature stores:** Feast (open-source), Tecton, AWS SageMaker Feature Store, Google Vertex Feature Store

---

## A/B Testing ML Models

Before fully deploying a new model:
1. **Shadow mode**: new model runs alongside old, predictions logged but not served — zero risk
2. **Canary deployment**: route a small % of traffic (5%) to new model
3. **A/B test**: randomly split traffic, measure business metrics, run statistical significance test
4. **Multi-armed bandit**: dynamically allocate more traffic to better-performing variant

**Statistical considerations:**
- Define success metric and minimum detectable effect upfront
- Calculate required sample size for statistical power
- Use z-test for proportions, t-test for continuous metrics
- Control for multiple testing (Bonferroni correction)

---

## LLM-Specific Production Concerns

### Latency
- Model size: 7B < 70B < 400B parameters
- Quantization: FP32 → FP16 → INT8 → INT4 (GPTQ, AWQ, llama.cpp)
- Speculative decoding: use a small "draft" model to propose tokens, verify with large model
- Continuous batching: (vLLM, TGI) batch variable-length inputs for maximum GPU utilization
- KV cache: avoid recomputing past tokens during generation

### Cost Optimization
- 1M tokens on GPT-4o: ~$5 (input) + ~$15 (output)
- 1M tokens on LLaMA via Groq: ~$0.59
- Self-hosting open models: higher upfront cost, cheaper at scale
- Caching: cache responses for repeated queries (semantic cache with embeddings)
- Prompt compression: remove redundant parts of the prompt

### Reliability
- Rate limiting and exponential backoff for API calls
- Fallback models when primary is unavailable
- Timeout handling — LLM calls can take 30+ seconds
- Circuit breakers to prevent cascade failures

### Security
- Prompt injection attacks: user input overrides system instructions
- PII leakage: LLMs can memorize and regurgitate training data
- Output validation: check for harmful content before returning to users
- API key rotation and least-privilege access

---

## RAG System Production Architecture

```
User
  ↓
API Gateway (rate limiting, auth)
  ↓
RAG API (FastAPI) ─── Cache (Redis)
  ├── Retriever
  │     ├── Vector DB (Qdrant/Pinecone)
  │     └── BM25 Index (Elasticsearch)
  ├── Reranker (Cohere API)
  └── Generator (LLM API)
        ↓
Monitoring (Evidently, Prometheus, Grafana)
```

### Scaling Considerations
- **Horizontal scaling**: run multiple API instances behind a load balancer
- **Vector DB**: use managed Qdrant Cloud, Pinecone, or Weaviate Cloud for HA
- **Caching layer**: Redis for often-asked questions (semantic cache)
- **Async processing**: use background tasks for slow operations (ingestion)
- **CDN**: serve static assets and cached responses from the edge

---

## Production Checklist for ML Systems

**Before deployment:**
- [ ] Model evaluated on holdout test set
- [ ] A/B test plan defined
- [ ] Rollback plan documented
- [ ] Load tested (can system handle peak traffic?)
- [ ] Monitoring dashboards configured
- [ ] Alerts for anomalies set up
- [ ] Documentation complete

**After deployment:**
- [ ] Monitor prediction distribution for first 24 hours
- [ ] Check latency SLAs (P99 < 2s)
- [ ] Verify business metrics are not harmed
- [ ] Schedule regular model retraining

---

## Popular MLOps Tools Summary

| Category | Tools |
|---|---|
| Experiment Tracking | MLflow, Weights & Biases, Neptune |
| Data Versioning | DVC, Delta Lake, Iceberg |
| Pipeline Orchestration | Airflow, Kubeflow, Prefect, ZenML |
| Model Serving | FastAPI, TorchServe, BentoML, Triton |
| Feature Store | Feast, Tecton, SageMaker FS |
| Monitoring | Evidently, Arize, WhyLogs, Grafana |
| Experiment/Model Registry | MLflow, Vertex AI, SageMaker |
| Container/Orchestration | Docker, Kubernetes, Helm |
