# Natural Language Processing (NLP) Fundamentals

## What is NLP?

Natural Language Processing (NLP) is the subfield of AI focused on enabling computers to understand, interpret, and generate human language. It bridges the gap between human communication and machine processing.

**Core NLP tasks:**
- Text Classification (sentiment, topic, spam)
- Named Entity Recognition (NER)
- Part-of-Speech Tagging (POS)
- Dependency Parsing
- Coreference Resolution
- Machine Translation
- Text Summarization
- Question Answering
- Information Extraction
- Language Generation

---

## Text Preprocessing

Before training any NLP model, raw text must be cleaned and normalized.

### Tokenization
Splitting text into tokens (words, subwords, or characters).

**Word tokenization**: split on whitespace and punctuation
- "Hello, world!" → ["Hello", ",", "world", "!"]
- Problems: "New York" becomes 2 tokens, contractions like "don't"

**Subword tokenization (BPE, WordPiece)**: break rare words into subwords
- "unhappiness" → ["un", "happiness"] or ["un", "##happi", "##ness"]
- Handles out-of-vocabulary words naturally

**Character tokenization**: single characters as tokens
- Smallest vocabulary (26 letters + symbols)
- Longest sequences — impractical for transformers

### Normalization
- **Lowercasing**: "Hello" → "hello"
- **Punctuation removal**: depends on task (keep for sentiment, remove for classification)
- **Stopword removal**: filter common words ("the", "a", "is") — useful for TF-IDF, not for neural models
- **Stemming**: reduce to stem word — "running" → "run" (rule-based, crude)
- **Lemmatization**: reduce to dictionary form — "better" → "good" (more accurate, requires POS tag)
- **Unicode normalization**: handle accents, special characters (NFKD, NFKC)

### Text Cleaning for RAG
- Remove HTML tags, markdown artifacts
- Normalize whitespace
- Handle headers, footers, page numbers
- Language detection and filtering
- Deduplication

---

## Classical NLP Representations

### Bag of Words (BoW)
Represent text as a vector of word counts, ignoring order:
- "The cat sat on the mat" → {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
- Vocabulary size defines feature dimensions
- Loses all word order and context

### TF-IDF (Term Frequency-Inverse Document Frequency)

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
TF(t, d) = count(t in d) / total_words(d)
IDF(t) = log(N / df(t))
```

Where N is total documents and df(t) is how many documents contain term t.
- Upweights rare, distinctive terms
- Downweights common terms that appear everywhere
- Standard baseline for information retrieval — still very competitive

### BM25 (Best Match 25)
Probabilistic extension of TF-IDF, the gold standard for keyword search:

```
BM25(d, Q) = Σ IDF(qᵢ) × [TF(qᵢ,d) × (k1+1)] / [TF(qᵢ,d) + k1×(1-b+b×|d|/avgdl)]
```

- k1 (1.2–2.0): controls term frequency saturation
- b (0.75): controls document length normalization
- **Used in Elasticsearch, Apache Lucene, this RAG system**

---

## Key NLP Tasks in Detail

### Named Entity Recognition (NER)
Identify and classify named entities in text:
- PERSON: "Elon Musk" 
- ORG: "OpenAI"
- LOC: "San Francisco"
- DATE: "March 2024"
- MONEY: "$1 billion"

**Approaches:** CRF (classical), BiLSTM-CRF, BERT-based sequence labeling (SoTA)

**Tools:** spaCy, Hugging Face Transformers, Flair

### Sentiment Analysis
Classify opinion polarity:
- Binary: positive / negative
- Multi-class: very negative / negative / neutral / positive / very positive
- Aspect-based: "The food was great but the service was terrible" → food: positive, service: negative

### Text Summarization

**Extractive**: select and concatenate important sentences from the original text
- TextRank (graph-based), BERT extractive summarizer
- No hallucination risk, but can be incoherent

**Abstractive**: generate new text that captures the key points
- BART, T5, Pegasus, LLMs
- More fluent but can hallucinate facts

### Machine Translation
- Statistical MT (phrase-based) → Google Translate pre-2016
- Neural MT (seq2seq with attention) → 2016
- Transformer-based (mBART, M2M-100, NLLB) → current SoTA
- GPT-4 level quality for high-resource language pairs

### Question Answering

**Extractive QA**: find the answer span in a provided passage (SQuAD benchmark)
- BERT fine-tuned on SQuAD was a major breakthrough

**Open-domain QA**: retrieve relevant documents then extract/generate answer
- DPR (Dense Passage Retrieval) + BERT reader
- Essentially a RAG pipeline!

**Closed-book QA**: LLM answers from parametric memory only (no retrieval)

### Text Classification
- Binary: spam detection, sentiment
- Multi-class: topic categorization, intent detection
- Multi-label: document tagging (can have multiple labels)

**Approaches:**
- Classical: TF-IDF + Logistic Regression / SVM (strong baseline!)
- Deep learning: LSTM, TextCNN
- Fine-tuned BERT: dominant approach for most tasks

---

## Evaluation Metrics for NLP

### Classification Metrics
- **Accuracy**: correct / total (misleading for imbalanced)
- **Precision**: TP / (TP + FP) — of predicted positives, how many were actually positive
- **Recall**: TP / (TP + FN) — of actual positives, how many did we find
- **F1**: harmonic mean of precision and recall
- **Macro F1**: average F1 per class, equal weight to each class
- **Micro F1**: pool all class predictions together (dominated by frequent classes)

### Sequence Labeling (NER, POS)
- **Token-level F1**: compare per-token labels
- **Entity-level F1**: exact span match required (used in CoNLL NER evaluation)

### Generation Metrics
- **BLEU** (Bilingual Evaluation Understudy): n-gram overlap with reference — standard for translation
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation): recall-focused n-gram overlap — standard for summarization
- **METEOR**: alignment-based metric, handles synonyms
- **BERTScore**: compute similarity using contextual BERT embeddings — better than BLEU/ROUGE
- **Human evaluation**: the gold standard — fluency, factuality, coherence ratings

---

## Information Extraction

### Relation Extraction
Identify semantic relationships between entities:
- "Elon Musk **founded** SpaceX" → (Elon Musk, founded, SpaceX)
- Used to build knowledge graphs

### Event Extraction
Identify events and their participants:
- "Apple announced a new iPhone on September 12" → EVENT: announce, AGENT: Apple, PRODUCT: iPhone, DATE: September 12

### Open Information Extraction (OpenIE)
Extract (subject, relation, object) triples without predefined relation types.
- Tools: Stanford OpenIE, AllenNLP

---

## NLP Libraries and Frameworks

### NLTK (Natural Language Toolkit)
The classic Python NLP library. Good for learning fundamentals (tokenization, POS tagging, parsing, corpora).
- Not state-of-the-art for most tasks
- Great for prototyping and educational purposes

### spaCy
Industrial-strength NLP library optimized for production:
- Fast tokenization, POS tagging, dependency parsing, NER
- Pre-trained models for 60+ languages
- Clean, efficient API

### Hugging Face Transformers
The go-to library for transformer models:
- Thousands of pre-trained models from the Hub
- Fine-tuning API, Trainer class, PEFT/LoRA support
- Works with PyTorch and TensorFlow

### Gensim
Specialized in topic modeling and document similarity:
- Word2Vec, FastText, GloVe training
- LDA (Latent Dirichlet Allocation) for topic modeling
- LSI (Latent Semantic Indexing)

### LangChain / LlamaIndex
Frameworks for building LLM-powered applications:
- Document loaders, text splitters, vector store integrations
- RAG pipeline abstractions, agent frameworks
- LlamaIndex focused on knowledge retrieval, LangChain more general

---

## Topic Modeling

Unsupervised method to discover abstract "topics" in a document collection.

### LDA (Latent Dirichlet Allocation)
Probabilistic model: each document is a mixture of topics, each topic is a distribution over words.
- Requires choosing K (number of topics) upfront
- Interpretable topics: {python, code, function, class} → programming

### NMF (Non-negative Matrix Factorization)
Decompose document-term matrix into two non-negative matrices (topics and document-topic weights).

### BERTopic
Modern approach using BERT embeddings + UMAP dimensionality reduction + HDBSCAN clustering:
- No need to specify K
- Topics are represented by key terms using c-TF-IDF
- Better coherence than LDA on short texts

---

## Common NLP Challenges

**Ambiguity**: "I saw the man with the telescope" — who had the telescope?
**Coreference**: "John went to the store. He bought milk." — He = John
**Sarcasm**: "Oh great, another Monday!" — positive words, negative sentiment
**Domain shift**: medical NLP model fails on legal documents
**Low-resource languages**: most research is English-centric
**Multilinguality**: handling code-switching, mixed-language text
**Long documents**: most models truncate at 512 or ~4K tokens
**Negation**: "The patient has no history of diabetes" — NER might still extract "diabetes"
