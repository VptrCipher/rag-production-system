# Retrieval-Augmented Generation (RAG): A Comprehensive Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by combining them with an external knowledge retrieval system. Instead of relying solely on the knowledge baked into a model's weights during training, RAG allows the LLM to dynamically access up-to-date, domain-specific, or proprietary information at query time.

The core idea is simple: before generating a response, the system retrieves the most relevant document chunks from a knowledge base and injects them as context into the prompt. This grounds the LLM's response in real evidence, dramatically reducing hallucinations and allowing the model to answer questions beyond its training cutoff.

## Why RAG?

Traditional LLMs have several limitations:
- **Knowledge cutoff**: They cannot answer questions about events after training.
- **Hallucination**: Without grounding, LLMs fabricate facts with high confidence.
- **No proprietary knowledge**: They cannot answer questions about internal company documents or databases.
- **High retraining cost**: Fine-tuning is expensive and slow.

RAG addresses all of these by externalising knowledge into a retrieval index that can be updated independently of the model.

## RAG Architecture

A standard RAG system has two main phases:

### 1. Indexing (Offline)
1. **Load** raw documents (PDF, HTML, Markdown, text)
2. **Chunk** documents into smaller, semantically coherent segments (typically 256–512 tokens)
3. **Embed** each chunk using a dense embedding model (e.g., OpenAI text-embedding-3-small or all-MiniLM-L6-v2)
4. **Store** chunk embeddings and metadata in a vector database (Qdrant, Pinecone, Weaviate, etc.)

### 2. Query-Time (Online)
1. **Embed** the user's query with the same model used at indexing time
2. **Retrieve** the top-K most similar chunks using Approximate Nearest Neighbour (ANN) search
3. **Rerank** candidates with a cross-encoder for higher precision
4. **Generate** a grounded response by injecting retrieved chunks into the LLM prompt

## Retrieval Strategies

### Dense Retrieval (Vector Search)
Dense retrieval encodes queries and documents into a continuous embedding space where semantically similar content clusters together. This excels at capturing conceptual similarity even without exact keyword overlap.

**Strengths**: Semantic understanding, handles paraphrasing well  
**Weaknesses**: Computationally expensive, struggles with rare/exact terms

### Sparse Retrieval (BM25)
BM25 is a probabilistic keyword matching algorithm. It scores documents based on term frequency (TF) and inverse document frequency (IDF). Despite its simplicity, BM25 remains competitive with dense retrieval for many queries.

**Strengths**: Fast, interpretable, great for exact keyword matches  
**Weaknesses**: No semantic understanding, synonym-blind

### Hybrid Retrieval
Hybrid retrieval combines dense and sparse results using techniques like Reciprocal Rank Fusion (RRF). RRF is robust to score distribution differences across retrievers:

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

Where `k` is a constant (typically 60) and `rank_i(d)` is the rank of document `d` in ranked list `i`.

**Strengths**: Best of both worlds, robust to edge cases  
**Weaknesses**: Slightly higher latency, requires tuning weights

## Chunking Strategies

The quality of chunking profoundly affects retrieval quality.

### Fixed-Size Chunking
Split documents into N-token chunks with M-token overlap. Simple and fast, but may cut sentences mid-thought.

### Sentence-Aware Chunking
Chunk on sentence boundaries to preserve semantic coherence. The preferred approach for most use cases.

### Recursive Character Splitting
Recursively split on paragraph → sentence → word boundaries until target size is reached.

### Semantic Chunking
Split based on embedding similarity between adjacent sentences — groups semantically coherent content together regardless of token count.

## Reranking

Initial retrieval (especially at large K) returns many noisy candidates. A reranker (cross-encoder) evaluates each (query, document) pair jointly, producing much more accurate relevance scores than bi-encoder retrieval.

**Popular rerankers**: Cohere Rerank, BGE Reranker, Jina Reranker

The typical pipeline is: retrieve top-30 candidates → rerank to top-5 → generate.

## Evaluation Metrics

### RAGAS Framework
RAGAS (Retrieval Augmented Generation Assessment) provides four core metrics:
- **Faithfulness**: Is the answer factually consistent with the retrieved context?
- **Answer Relevancy**: Does the answer actually address the question?
- **Context Precision**: What fraction of retrieved chunks are relevant?
- **Context Recall**: What fraction of required information was retrieved?

### Other Metrics
- **Mean Reciprocal Rank (MRR)**: How early does the first relevant result appear?
- **NDCG@K**: Normalised Discounted Cumulative Gain — measures ranking quality
- **Latency**: End-to-end query latency (p50, p95, p99)

## Advanced RAG Techniques

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer to the query, embed it, and use that embedding for retrieval. This helps bridge the gap between short queries and long documents.

### Query Decomposition
Break complex multi-hop queries into simpler sub-queries, retrieve for each, then synthesise a final answer.

### Multi-Vector Representation
Store multiple embeddings per document chunk (summary + full text) to capture different aspects.

### RAG Fusion
Generate multiple paraphrased queries, retrieve for each, and fuse the results. Increases coverage diversity.

### Contextual Compression
After retrieval, extract only the most relevant sentences from each chunk before injecting into the LLM prompt.

## Production Considerations

### Latency Budget
- Embedding query: ~5–15ms (local model) or ~20–50ms (API)
- Vector search: ~10–50ms for 1M vectors
- Reranking 30 docs: ~200–400ms (Cohere API)
- LLM generation: ~500ms–3s (GPT-4o)

### Cost Management
- Cache frequent query embeddings
- Use local embedding models (all-MiniLM-L6-v2) to eliminate embedding API costs
- Tune top-K to minimum needed — fewer reranking calls = lower cost

### Monitoring
- Track retrieval quality metrics over time
- Alert on latency regressions
- Log all queries, retrieved chunks, and generated answers for audit
