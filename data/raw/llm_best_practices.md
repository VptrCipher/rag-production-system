# LLM Best Practices for Production RAG Systems

## Prompt Engineering

### The Grounded Response Pattern
The most important prompt engineering principle for RAG systems is grounding — forcing the LLM to base its answer only on retrieved context, not on its parametric knowledge.

**Effective grounding prompt structure**:
```
You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Cite your sources as [Source N] for every factual claim.

CONTEXT:
[Source 1] (from document_a.pdf, chunk 3): ...
[Source 2] (from document_b.md, chunk 7): ...

QUESTION: {user_question}

ANSWER:
```

### Why Citation Is Critical
Requiring `[Source N]` citations in every factual claim:
1. Forces the model to stay grounded in retrieved context
2. Makes faithfulness evaluation tractable (automated checking)
3. Allows users to verify claims against primary sources
4. Reduces hallucination dramatically compared to uncited prompts

### System vs User Messages
Always separate the grounding instruction (system message) from the context + query (user message). This prevents the model from ignoring instructions buried in a long context block.

```python
messages = [
    {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
    {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
]
```

## Temperature Tuning

| Use Case | Temperature | Rationale |
|---|---|---|
| Factual Q&A (RAG) | 0.0 – 0.1 | Deterministic, grounded in context |
| Summarisation | 0.1 – 0.3 | Slight variation for readability |
| Creative writing | 0.7 – 1.0 | High diversity encouraged |
| Code generation | 0.0 – 0.2 | Correctness over creativity |

For production RAG, **temperature 0.1** is the recommended sweet spot — near-deterministic but not rigidly templated.

## Hallucination Reduction Strategies

### 1. Strict Grounding Prompts
Explicitly prohibit out-of-context answers. Use phrases like:
- "You MUST only use information from the provided context."
- "If you are not sure, say 'I don't know'. Do not guess."
- "Do not use any information from your training data."

### 2. Retrieval Quality
Better retrieval = less hallucination. Every hallucination in a RAG system is a retrieval failure — the correct context wasn't retrieved, so the LLM filled the gap with generated content.

Improve retrieval quality by:
- **Increasing top-K**: Retrieve more candidates before reranking
- **Using hybrid search**: BM25 catches keyword-specific queries that vector search misses
- **Reranking**: Cohere/BGE rerankers dramatically improve precision at top-5
- **Chunking strategy**: Sentence-level chunks preserve semantic coherence better than fixed-token splits

### 3. Context Length Management
When retrieved context exceeds the model's optimal attention window (~4K tokens for most models), later chunks get less attention. Strategies:
- **Truncate context** to the most relevant chunks (top-3 to top-5 after reranking)
- **Compress**: Extract only the most relevant sentences from each chunk
- **Flash Attention**: Use models supporting very long contexts (GPT-4-128k, Claude 3)

### 4. Self-Consistency Checking
Generate answer N times with different random seeds, then return the majority answer. Works well for factual questions but increases latency and cost N×.

### 5. Answer Validation
After generation, use a second LLM call to verify whether the answer contradicts the source context. Useful for high-stakes applications.

## Token Efficiency

### Max Tokens Budget
For a GPT-4o call with 128k context window:
- System prompt: ~200 tokens
- Context (5 chunks × 512 tokens): ~2,560 tokens
- Query: ~50 tokens
- Generation budget: 1,024 tokens
- **Total**: ~3,834 tokens per query

### Reducing Prompt Cost
1. **Compress context**: Use extractive summarisation to shorten chunks before injection
2. **Cache repeating prefixes**: Use OpenAI Prompt Caching for frequently reused system prompts
3. **Use smaller models for simple queries**: Route to GPT-3.5 or GPT-4o-mini when confidence is high

## Output Formatting

### Structured Output
Use `response_format={"type": "json_object"}` for programmatic downstream processing:
```python
completion = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "Extract entities: ..."}],
)
```

### Markdown Output
For user-facing chat applications, instruct the model to format answers in Markdown for better readability in chat UIs.

## Streaming for Better UX

For interactive applications, always use streaming to display partial responses as they are generated:
```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

This reduces perceived latency from 2+ seconds to near-instant first token.

## Error Handling and Retries

### Transient API Errors
Always wrap LLM calls with exponential backoff:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def call_llm(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Rate Limiting
- Track token usage per minute and implement a token bucket
- Use `tiktoken` to pre-count tokens before sending requests
- Back off with 429 responses — use `retry-after` header if present

### Fallback Models
If the primary model (GPT-4o) fails or is rate-limited, fall back to GPT-4o-mini or a local Ollama model. Ensure your response interface is model-agnostic.

## Observability

### What to Log
Every RAG query should log:
- Input query
- Retrieved chunk IDs and scores
- Reranked chunk IDs and scores
- Final context injected into prompt
- LLM model used
- Token counts (prompt, completion, total)
- End-to-end latency
- Generated answer

### Evaluation in Production
Run continuous RAGAS evaluation on a random sample of production queries using golden reference answers. Alert when faithfulness drops below 0.75 or answer relevancy drops below 0.70.

## Security Considerations

### Prompt Injection
Malicious content in retrieved documents can hijack the LLM's instructions. Mitigations:
- Use clear structural separators between system prompt, context, and query
- Never allow retrieved content to appear in the system message
- Consider output validation to detect prompt injection attempts

### Data Privacy
- Never log full document content if it contains PII
- Use document-level access controls — only retrieve documents the user is authorised to see
- Consider field-level encryption for sensitive metadata payloads in Qdrant
