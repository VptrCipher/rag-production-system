---
title: RAG-Production-System
emoji: 🚀
colorFrom: green
colorTo: black
sdk: docker
pinned: false
---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=venom&color=0:000000,50:003300,100:000000&height=200&section=header&text=VptrCipher-RAG&fontSize=60&fontColor=00ff41&animation=fadeIn&fontAlignY=65&desc=//%20LLAMAINDEX%20+%20HF%20EMBEDDINGS%20+%20GROQ%20+%20QDRANT&descAlignY=85&descAlign=50&descSize=18&descColor=00ff41" width="100%"/>
</p>

<!-- ANIMATED TYPING -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Share+Tech+Mono&size=22&duration=2000&pause=500&color=00FF41&background=000000&center=true&vCenter=true&multiline=false&width=600&lines=%5B+SYSTEM+BOOT+%5D+Initializing+VptrCipher+RAG;%5B+AUTH+%5D+HuggingFace+Local+Embeddings;%5B+MODE+%5D+HYBRID+RETRIEVAL+ACTIVE;%5B+STATUS+%5D+Deployed+on+FastAPI+%2B+Docker;%5B+WARNING+%5D+Optimized+for+Production+%E2%9A%A0%EF%B8%8F" />
</p>

<!-- HACKER DIVIDER -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## `> systemctl status rag-engine.service`

```bash
● rag-production-system.service - AI Retrieval-Augmented Generation Engine
     Status: Operational (Standard HF Spaces Port: 7860)
     Engine: LlamaIndex @ v0.10+
     Vector Store: Qdrant (Local Persistent Storage)
     Embeddings: HuggingFace Local (BGE-Small-En-v1.5)
     Inference: Groq (Llama 3.3 70B) / OpenAI (GPT-4o)
     Observability: Arize Phoenix @ Port 6006
```

---

## `> architecture --verify`

```mermaid
graph TD
    A[User Request] --> B[FastAPI Gateway]
    B --> C[Semantic Chunking / Ingest]
    C --> D[Embedding Model - HuggingFace Local]
    D --> E{Hybrid Retrieval}
    E -->|Dense| F[Qdrant Vector DB]
    E -->|Sparse| G[Rank-BM25 Keyword]
    F --> H[Cohere Re-Ranking]
    G --> H
    H --> I[Grounded Generation - Groq/OpenAI]
    I --> J[Streaming Response with Citations]
    J --> K[Arize Phoenix Tracing]
    K --> B
```

---

## `> features --all`

- **🧬 Semantic Chunking**: Intelligent splitting for maximum context retention via `SemanticChunker`.
- **⚡ Hybrid Retrieval**: Dense Vector (Qdrant) + Sparse BM25 for 99% keyword/semantic recall.
- **🤗 HuggingFace Native**: Uses local HuggingFace embeddings for high-speed, cost-effective indexing.
- **🎯 Precise Re-Ranking**: Cohere Rerank layer ensures only the top-N most relevant chunks feed the LLM.
- **🔥 Citations & Grounding**: Strict system prompts force the model to cite `[Source N]` for every claim.
- **🔍 Full Observability**: Integrated Arize Phoenix for OpenInference tracing and RAGAS evaluation.
- **📦 Cloud Ready**: Native Docker support for seamless deployment to **Hugging Face Spaces**.

---

## `> ls /tech-stack`

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,fastapi,docker,pytorch,huggingface,firebase,github,linux&theme=dark" />
</p>

- **Core**: LlamaIndex, Transformers, Sentence-Transformers
- **Vector DB**: Qdrant (Local / Server)
- **Inference**: Groq (Llama 3.3), OpenAI API
- **Evaluation**: Ragas, Datasets
- **Monitoring**: Arize Phoenix, OTel

---

## `> deploy --target huggingface`

This project is optimized for **Hugging Face Spaces** using the provided `Dockerfile`.

### Automated Deployment
1. Create a new **Docker Space** on Hugging Face.
2. Push this repository to the Space.
3. HF will automatically build the image and expose the API/UI on port **7860**.

### Required Environment Variables
Configure these in your HF Settings > Variables:
- `GROQ_API_KEY`: For Llama 3 generation.
- `COHERE_API_KEY`: For reranking layer.
- `OPENAI_API_KEY`: (Optional) For GPT-4o paths.

---

## `> setup --local`

### 1. Initialize
```bash
git clone https://github.com/VptrCipher/rag-production-system.git
cd rag-production-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Required: GROQ_API_KEY, COHERE_API_KEY
```

### 3. Run
```bash
python start_all.py
```

- **API**: `http://localhost:8000/api/v1`
- **Interface**: `http://localhost:8000/`
- **Tracing**: `http://localhost:6006`

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Share+Tech+Mono&size=13&duration=3000&pause=500&color=00FF41&center=true&vCenter=true&width=500&lines=//+VptrCipher+Production+System;//+Knowledge+is+Weaponized+Context.;//+Stay+Dangerous.+Ship+Fast." />
</p>
