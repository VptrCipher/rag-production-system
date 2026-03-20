# Embeddings and Semantic Search

## What are Embeddings?

Embeddings are dense numerical representations (vectors) of data — text, images, audio, or structured data — that encode semantic meaning in a high-dimensional space. Two semantically similar items will have vectors that are close together.

**Key properties:**
- **Dense**: every dimension carries information (vs. sparse TF-IDF vectors)
- **Fixed-size**: regardless of input length
- **Semantic**: similar meaning → similar vector
- **Transferable**: learned on large data, useful for many downstream tasks

---

## Text Embedding Models

### Word2Vec (2013)
Shallow neural network trained to predict:
- **CBOW**: predict center word from surrounding context
- **Skip-gram**: predict context words from center word

Famous property: `king - man + woman ≈ queen`

Limitations: single vector per word (no disambiguation), no subword handling.

### GloVe (Global Vectors)
Matrix factorization on word co-occurrence statistics. Faster to train than Word2Vec, similar quality.

### FastText
Extension of Word2Vec using character n-grams. Can represent out-of-vocabulary words by composing subword vectors. Works well for morphologically rich languages.

### ELMo (Embeddings from Language Models)
Bidirectional LSTM-based contextual embeddings. Unlike Word2Vec, the representation of "bank" changes depending on context.

### BERT Embeddings
Take [CLS] token representation or average pool all token representations from a BERT model. Rich contextual embeddings but originally not optimized for similarity tasks.

### Sentence Transformers (SBERT)
Fine-tuned BERT with a Siamese network architecture using contrastive loss (e.g., NLI datasets). Produces sentence-level embeddings optimized for semantic similarity.

**Popular SBERT models:**
- `all-MiniLM-L6-v2` — 384 dimensions, fast, great for most use cases (**used in this system**)
- `all-mpnet-base-v2` — 768 dimensions, better quality
- `multi-qa-mpnet-base-dot-v1` — optimized for question answering
- `paraphrase-multilingual-mpnet-base-v2` — multilingual support

### OpenAI Embeddings
- `text-embedding-3-small` — 1536 dimensions, cheap, good quality
- `text-embedding-3-large` — 3072 dimensions, best quality, higher cost

### Cohere Embeddings
- `embed-english-v3.0` — 1024 dimensions, excellent quality
- `embed-multilingual-v3.0` — multilingual support

---

## Similarity Metrics

### Cosine Similarity
Measures the angle between two vectors, ignoring magnitude:
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
- **Most common for NLP** — works even if vectors have different magnitudes
- If vectors are L2-normalized, cosine similarity = dot product

### Dot Product
Fast to compute, magnitude-sensitive. Preferred when you want larger, "louder" vectors to score higher. Used in many recommendation systems.

### Euclidean Distance (L2)
```
d(A, B) = sqrt(Σ(Aᵢ - Bᵢ)²)
```
Intuitive geometric distance. Less common for text embeddings.

### Which to Use?
- **Cosine**: normalized embeddings, text similarity tasks
- **Dot product**: recommendation systems, when magnitude matters
- **L2**: image embeddings, explicit geometric interpretation needed

---

## Approximate Nearest Neighbor (ANN) Search

Exact k-NN is O(n × d) — too slow for millions of vectors. ANN algorithms trade a small accuracy loss for massive speed gains.

### HNSW (Hierarchical Navigable Small World)
- Graph-based structure with multiple layers
- Layer 0: all nodes; higher layers: progressively sparser
- Navigate from sparse upper layers to dense lower layers
- **Best recall/speed tradeoff** for most use cases
- Used by: Qdrant, Weaviate, Milvus

### IVF (Inverted File Index)
- Cluster vectors using K-Means
- At query time, search only the closest clusters (nprobe parameter)
- Used by: FAISS

### LSH (Locality Sensitive Hashing)
Hash similar vectors to the same bucket. Fast but lower recall than HNSW.

### ScaNN (Google)
Anisotropic vector quantization. Extremely fast on Google hardware.

---

## Embedding Models Performance Benchmarks (MTEB Leaderboard)

The Massive Text Embedding Benchmark (MTEB) evaluates models across 56 tasks in 8 categories:

| Model | Avg Score | Params | Dim |
|---|---|---|---|
| `text-embedding-3-large` | 64.6 | — | 3072 |
| `cohere-embed-v3` | 64.5 | — | 1024 |
| `e5-large-v2` | 62.3 | 335M | 1024 |
| `all-mpnet-base-v2` | 57.8 | 109M | 768 |
| `all-MiniLM-L6-v2` | 56.3 | 22M | 384 |

Smaller models like MiniLM are preferred when latency is critical.

---

## How Embedding Models are Trained

### Contrastive Learning
The most common approach. Pairs of semantically similar examples should have similar embeddings; dissimilar pairs should be far apart.

**Loss functions:**
- **Triplet Loss**: anchor, positive, negative — push negative away, pull positive close
- **InfoNCE / NT-Xent**: normalized temperature-scaled cross entropy (used in SimCSE, CLIP)
- **MNRL (Multiple Negatives Ranking Loss)**: efficiently uses all in-batch negatives

### Training Data Sources
- NLI pairs (SNLI, MultiNLI): entailment pairs as positives, contradictions as negatives
- MS MARCO: query-passage pairs from Bing search
- MEDI (2022): 330 datasets combined for diverse training

### Matryoshka Representation Learning (MRL)
Trains embeddings that work at multiple dimensions:
- A 1536-dim vector can be truncated to 768, 384, or 256 dims and still be effective
- Used by OpenAI's `text-embedding-3-*` models
- Trade Quality for Speed by varying the dimension at inference time

---

## Semantic Search Pipeline

### Step 1: Offline Indexing
```
Documents → Chunking → Embedding → Vector Store
```

### Step 2: Online Retrieval
```
Query → Query Embedding → ANN Search → Top-K Results
```

### Chunking Strategies
**Fixed-size chunking**: split every N tokens with overlap
- Simple, predictable
- Risk: splits mid-sentence or mid-concept

**Sentence-aware chunking**: respect sentence boundaries
- Better semantic coherence

**Recursive chunking**: try to split on paragraphs → sentences → words
- Used by LangChain's RecursiveCharacterTextSplitter

**Semantic chunking**: group sentences that are semantically similar
- More expensive, better results

**Typical parameters**: 256–512 tokens per chunk, 10–20% overlap

---

## Dense vs. Sparse Retrieval

### Dense Retrieval (semantic)
- Embedding-based, finds conceptually similar documents
- Handles paraphrases, synonyms, multilingual queries
- Requires embedding model (compute cost at index and query time)
- Example: bi-encoder (SBERT) + FAISS/Qdrant

### Sparse Retrieval (lexical)
- Keyword-based (TF-IDF, BM25)
- Fast, no model needed
- Fails on synonyms ("automobile" vs "car")
- Great for exact term matches, codes, product IDs

### Hybrid Retrieval (best of both)
Combine dense + sparse retrieval scores using:
- **Reciprocal Rank Fusion (RRF)**: merge ranked lists without needing to normalize scores
- **Linear combination**: `score = α × dense_score + (1-α) × sparse_score`

**RRF formula:**
```
RRF_score(d) = Σ 1 / (k + rank(d))
```
where k=60 is a constant that reduces the impact of very high ranks.

---

## Cross-Encoders for Reranking

Bi-encoders are fast but less accurate (each document is encoded independently).
Cross-encoders jointly encode the query + document and produce a more accurate relevance score.

```
Cross-Encoder([query, document]) → relevance score
```

**Trade-off:** 
- Cross-encoders: high accuracy, very slow (O(n) model calls per query)
- Bi-encoders: fast (pre-index), slightly lower accuracy

**Typical pipeline:**
1. Bi-encoder retrieves top-100 candidates quickly
2. Cross-encoder reranks top-100 to get final top-5

**Popular cross-encoder rerankers:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `Cohere rerank-english-v3.0` (used in this system)
- `Jina Reranker`
- `BGE Reranker`

---

## Multimodal Embeddings

### CLIP (OpenAI, 2021)
Trains image encoder + text encoder contrastively on 400M image-text pairs.
- Images and text are embedded in the same space
- Zero-shot image classification, image-text retrieval

### DALL-E / Stable Diffusion
Use CLIP embeddings as conditioning for image generation.

### ImageBind (Meta, 2023)
Embeds 6 modalities (images, text, audio, depth, thermal, IMU) in a shared space.

---

## Evaluating Embedding Quality

- **BEIR Benchmark**: heterogeneous retrieval tasks (biomedical, financial, etc.)
- **MTEB**: 56 tasks including retrieval, classification, clustering, summarization
- **NanoBEIR**: fast version for development evaluation
- **Domain-specific evaluation**: always test on your actual data distribution
