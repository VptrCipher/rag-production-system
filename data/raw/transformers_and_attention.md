# Transformers and Attention Mechanisms

## Introduction

The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al., revolutionized natural language processing and has become the backbone of modern AI systems including GPT-4, BERT, LLaMA, Claude, and Gemini.

Unlike RNNs and LSTMs, Transformers process all tokens in parallel and use attention to relate every position in a sequence to every other position — regardless of distance.

---

## The Attention Mechanism

### Core Intuition
Attention allows the model to focus on relevant parts of the input when producing each output. For the word "bank" in "I went to the bank to deposit money," attention helps the model understand it refers to a financial institution, not a riverbank, by attending to nearby context.

### Scaled Dot-Product Attention

Given three matrices:
- **Q (Query)**: what we're looking for
- **K (Key)**: what each token offers
- **V (Value)**: the actual content to aggregate

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

The scaling factor `√d_k` prevents dot products from growing too large in high dimensions, keeping gradients stable.

### Multi-Head Attention
Instead of a single attention function, run h attention heads in parallel with different learned projections:
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W_O
```

Each head learns to attend to different aspects of the input (syntax, semantics, coreference, etc.).

---

## Transformer Architecture

### Encoder
Processes the entire input sequence and produces contextualized representations.

**Each encoder layer contains:**
1. Multi-Head Self-Attention
2. Add & Normalize (residual connection + layer norm)
3. Position-wise Feed-Forward Network (FFN)
4. Add & Normalize

**Self-attention** allows each token to attend to all other tokens in the input.

### Decoder
Generates the output sequence token by token.

**Each decoder layer contains:**
1. Masked Multi-Head Self-Attention (causal — can only see past tokens)
2. Add & Normalize
3. Cross-Attention (attends to the encoder output)
4. Add & Normalize
5. Feed-Forward Network
6. Add & Normalize

### Positional Encoding
Transformers have no inherent notion of order. Positional encodings are added to embeddings:
- **Sinusoidal (original)**: fixed encodings using sine/cosine at different frequencies
- **Learned positional embeddings**: trained alongside the model (BERT)
- **Rotary Positional Embeddings (RoPE)**: encode relative positions, used in LLaMA and GPT-NeoX
- **ALiBi**: attention with linear biases, allows extrapolation to longer sequences

---

## BERT (Bidirectional Encoder Representations from Transformers)

BERT is an encoder-only Transformer, pre-trained with:
1. **Masked Language Modeling (MLM)**: predict randomly masked tokens using both left and right context
2. **Next Sentence Prediction (NSP)**: predict whether two sentences follow each other

**Key properties:**
- Bidirectional — sees full context in both directions
- Pre-trained then fine-tuned for downstream tasks
- Variants: RoBERTa (removes NSP, more data), ALBERT, DistilBERT, DeBERTa

**Best for:** Classification, NER, question answering, semantic similarity

---

## GPT (Generative Pre-trained Transformer)

GPT is a decoder-only Transformer, pre-trained with:
- **Next Token Prediction (Causal LM)**: predict each next token given all previous tokens

**Key properties:**
- Autoregressive — generates one token at a time
- Scales exceptionally well (GPT-3: 175B params, GPT-4: ~1T)
- In-context learning: learns tasks from a few examples in the prompt without weight updates

**GPT family**: GPT-1 → GPT-2 → GPT-3 → InstructGPT → ChatGPT → GPT-4

---

## T5 (Text-to-Text Transfer Transformer)

Encoder-decoder Transformer where every NLP task is framed as text-to-text:
- Translation: "translate English to German: Hello" → "Hallo"
- Summarization: "summarize: ..." → summary
- Classification: "classify: positive or negative: Great movie!" → "positive"

**Advantage:** Unified interface for all tasks. Variants: T5-small, T5-base, T5-large, T5-11B, Flan-T5

---

## LLaMA (Large Language Model Meta AI)

Open-weight decoder-only LLM family from Meta:
- LLaMA 1 (7B–65B), LLaMA 2 (7B–70B), LLaMA 3 (8B–70B), LLaMA 3.3 (70B)
- Uses RoPE for positional encoding, SwiGLU activation, RMSNorm
- Fine-tuned variants: Alpaca, Vicuna, Mistral (from Mistral AI)

**LLaMA 3.3-70b-versatile** (used in this RAG system via Groq):
- State-of-the-art open-weight model at 70B parameters
- Strong reasoning, coding, and instruction-following capabilities
- Available via Groq's inference API at very low latency

---

## Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (2022 — "Chinchilla") showed:
- Model performance scales predictably with model size, data size, and compute
- **Chinchilla optimal**: for a given compute budget, train a smaller model on more data
  - Example: LLaMA-7B trained on 1T tokens outperforms GPT-3 (175B) trained on 300B tokens

---

## Key Architectural Improvements

### Flash Attention
Optimized attention computation that reduces memory from O(N²) to O(N) by using tiling, avoiding materializing the full attention matrix. Enables much longer context windows.

### Grouped Query Attention (GQA)
Multiple query heads share a single key-value head. Used in LLaMA 3, Mistral. Reduces memory and inference cost.

### Mixture of Experts (MoE)
Only activates a subset of parameters per token (sparse activation). Used in GPT-4, Mixtral.
- `total params >> active params per token`
- Bigger model capacity at same compute cost

### KV Cache
During autoregressive generation, past key-value pairs are cached so they don't need to be recomputed. Critical for inference efficiency.

---

## Fine-Tuning Techniques

### Full Fine-Tuning
Update all model weights on task-specific data. Expensive but most powerful.

### LoRA (Low-Rank Adaptation)
Freeze the original weights and add small trainable low-rank matrices:
- `W_new = W_original + ΔW = W_original + A × B`
- A and B are small matrices with rank r << d
- Only A and B are trained — drastically fewer trainable parameters
- QLoRA: quantize the frozen weights to 4-bit, train LoRA adapters in bf16

### RLHF (Reinforcement Learning from Human Feedback)
Fine-tune LLMs to follow instructions and be helpful:
1. Supervised Fine-Tuning (SFT) on demonstrations
2. Train a Reward Model on human preference rankings
3. Use PPO to optimize the LLM against the reward model

Used by: InstructGPT, ChatGPT, Claude, Gemini

### DPO (Direct Preference Optimization)
A simpler alternative to RLHF that directly fine-tunes on preference data without a separate reward model.

---

## Context Window and Long Context

- GPT-4: 128K tokens
- Claude 3.5: 200K tokens
- Gemini 1.5 Pro: 1M tokens

**Challenges with long context:**
- Attention is O(N²) in sequence length (mitigated by Flash Attention)
- "Lost in the middle" problem: models struggle with information in the middle of long contexts
- Position interpolation: extend context beyond training length

---

## Tokenization

Transformers operate on tokens, not characters or words.

- **BPE (Byte-Pair Encoding)**: iteratively merges frequent character pairs. Used by GPT
- **WordPiece**: similar to BPE, used by BERT
- **SentencePiece**: language-agnostic, works at byte level, used by T5, LLaMA
- Typical vocabulary: 32K–100K tokens

Common token approximations:
- 1 token ≈ 4 characters in English
- 100 tokens ≈ 75 words
- 1 page ≈ 500 tokens
