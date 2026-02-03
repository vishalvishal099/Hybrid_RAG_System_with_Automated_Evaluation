# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
│                       (Streamlit App - app.py)                       │
│  - Query Input                                                       │
│  - Answer Display                                                    │
│  - Source Visualization                                              │
│  - Metrics Dashboard                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     HYBRID RAG SYSTEM                                │
│                   (src/rag_system.py)                                │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    QUERY PROCESSING                          │   │
│  │  - Tokenization                                              │   │
│  │  - Preprocessing                                             │   │
│  └────────────────────┬─────────────────────────────────────────┘   │
│                       │                                              │
│         ┌─────────────┴─────────────┐                                │
│         ▼                           ▼                                │
│  ┌─────────────────┐         ┌─────────────────┐                    │
│  │ Dense Retrieval │         │Sparse Retrieval │                    │
│  │   (FAISS)       │         │    (BM25)       │                    │
│  │                 │         │                 │                    │
│  │ - Embeddings    │         │ - Tokenization  │                    │
│  │ - Vector Search │         │ - TF-IDF        │                    │
│  │ - Top-K (20)    │         │ - Top-K (20)    │                    │
│  └────────┬────────┘         └────────┬────────┘                    │
│           │                           │                              │
│           └─────────────┬─────────────┘                              │
│                         ▼                                            │
│            ┌────────────────────────┐                                │
│            │Reciprocal Rank Fusion │                                │
│            │       (RRF)            │                                │
│            │                        │                                │
│            │ - Combine Rankings     │                                │
│            │ - Calculate RRF Score  │                                │
│            │ - Top-N (5) Chunks     │                                │
│            └────────┬───────────────┘                                │
│                     │                                                │
│                     ▼                                                │
│         ┌───────────────────────┐                                    │
│         │  Context Assembly     │                                    │
│         │  - Concatenate Chunks │                                    │
│         │  - Add Source Info    │                                    │
│         │  - Create Prompt      │                                    │
│         └───────────┬───────────┘                                    │
│                     │                                                │
│                     ▼                                                │
│         ┌───────────────────────┐                                    │
│         │  LLM Generation       │                                    │
│         │  (Flan-T5)            │                                    │
│         │  - Context-aware      │                                    │
│         │  - Answer Generation  │                                    │
│         └───────────┬───────────┘                                    │
│                     │                                                │
└─────────────────────┼────────────────────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │     RESPONSE     │
            │  - Answer        │
            │  - Sources       │
            │  - Scores        │
            │  - Timing        │
            └─────────────────┘
```

## Data Flow

```
Wikipedia URLs (500)
    │
    ├─ Fixed (200) ──────────┐
    │                        │
    └─ Random (300) ─────────┤
                             │
                             ▼
                  ┌──────────────────┐
                  │ Data Collection  │
                  │ - Scraping       │
                  │ - Cleaning       │
                  │ - Chunking       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │     Corpus       │
                  │  (corpus.json)   │
                  │  - Documents     │
                  │  - Chunks        │
                  │  - Metadata      │
                  └────────┬─────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    ┌───────────────┐           ┌───────────────┐
    │Dense Indexing │           │Sparse Indexing│
    │  - Embed all  │           │  - Tokenize   │
    │  - Build FAISS│           │  - Build BM25 │
    └───────┬───────┘           └───────┬───────┘
            │                           │
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │ faiss_index   │           │bm25_index.pkl │
    └───────────────┘           └───────────────┘
```

## Retrieval Pipeline

```
                        User Query
                            │
                            ▼
                ┌─────────────────────┐
                │   Preprocessing     │
                └───────┬─────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐               ┌───────────────┐
│Dense Retrieval│               │Sparse Retrieval│
│               │               │                │
│Query → Embed  │               │Query → Tokens  │
│FAISS Search   │               │BM25 Scoring    │
│Top-20 Chunks  │               │Top-20 Chunks   │
└───────┬───────┘               └───────┬────────┘
        │                               │
        │   Ranks: [1,2,3,...,20]      │   Ranks: [1,2,3,...,20]
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │         RRF           │
            │  Score = Σ 1/(60+rank)│
            │  Merge & Re-rank      │
            └───────┬───────────────┘
                    │
                    ▼
            Top-5 Chunks (by RRF)
                    │
                    ▼
            ┌───────────────┐
            │ Context Pool  │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  LLM (Flan-T5)│
            └───────┬───────┘
                    │
                    ▼
                  Answer
```

## Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION PIPELINE                         │
│                (evaluation/pipeline.py)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Load 100 Questions            │
        │ (questions_100.json)          │
        └───────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────────────┐
        │ For Each Question:            │
        │ 1. Run RAG System             │
        │ 2. Get Response               │
        │ 3. Evaluate Metrics           │
        └───────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐   ┌───────────────────┐
│Basic Metrics  │   │ Advanced Features │
│- MRR          │   │- Ablation Study   │
│- NDCG@5       │   │- Error Analysis   │
│- BERTScore    │   │- LLM-as-Judge     │
│- Precision    │   │- Adversarial Test │
│- Recall       │   │- Calibration      │
└───────┬───────┘   └───────┬───────────┘
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Aggregate      │
        │  Results        │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Generate       │
        │  Reports        │
        │  - JSON         │
        │  - CSV          │
        │  - Charts       │
        └─────────────────┘
```

## Module Dependencies

```
app.py (Streamlit UI)
    │
    └─► src/rag_system.py (Core RAG)
            │
            ├─► sentence_transformers (Embeddings)
            ├─► faiss (Vector Search)
            ├─► rank_bm25 (Sparse Retrieval)
            └─► transformers (LLM)

evaluation/pipeline.py (Evaluation)
    │
    ├─► evaluation/metrics.py
    │       │
    │       ├─► bert_score
    │       ├─► rouge_score
    │       └─► sklearn
    │
    ├─► evaluation/innovative_eval.py
    │       │
    │       └─► matplotlib, seaborn
    │
    └─► src/rag_system.py

src/data_collection.py
    │
    ├─► wikipediaapi (Data Fetching)
    ├─► beautifulsoup4 (Parsing)
    └─► tiktoken (Token Counting)

src/question_generation.py
    │
    └─► transformers (Question Gen)
```

## Storage Layout

```
Data Storage:
├── corpus.json (~500MB)
│   ├── metadata
│   ├── documents[]
│   └── chunks[]
│
├── faiss_index (~150MB)
│   └── Vector embeddings
│
└── bm25_index.pkl (~80MB)
    └── Inverted index

Model Cache:
├── ~/.cache/huggingface/
│   ├── sentence-transformers/
│   │   └── all-MiniLM-L6-v2/
│   └── transformers/
│       └── google/flan-t5-base/
```

## Request Flow (Example)

```
1. User Query: "What is quantum computing?"
   │
2. ┌─ Dense: Embed query → FAISS search
   │  Results: [doc_5, doc_23, doc_67, ...]
   │  Scores: [0.85, 0.79, 0.73, ...]
   │
   └─ Sparse: Tokenize query → BM25 scoring
      Results: [doc_23, doc_5, doc_89, ...]
      Scores: [12.3, 10.8, 9.2, ...]
   
3. RRF Fusion:
   doc_5:  1/(60+1) + 1/(60+2) = 0.0325
   doc_23: 1/(60+2) + 1/(60+1) = 0.0325
   doc_67: 1/(60+3) + 0 = 0.0159
   ...
   Ranked: [doc_5, doc_23, doc_67, doc_89, doc_12]

4. Context Assembly:
   "[Source: Quantum Computing]
    Quantum computing is...
    
    [Source: Computer Science]
    Related to quantum physics..."

5. LLM Generation:
   Input: Context + Query
   Output: "Quantum computing is a type of 
           computation that harnesses quantum 
           mechanics to process information..."

6. Response:
   - Answer: "Quantum computing is..."
   - Sources: [Quantum Computing, Computer Science, ...]
   - Scores: {dense: 0.85, sparse: 12.3, rrf: 0.0325}
   - Time: {retrieval: 0.12s, generation: 1.45s}
```

---

**Legend**:
- `│` : Data flow
- `►` : Dependency
- `[]` : Array/List
- `{}` : Dictionary/Object

**File Sizes** (Approximate):
- Corpus: 500MB
- FAISS Index: 150MB
- BM25 Index: 80MB
- Models (cached): 2-5GB

**Processing Times** (Approximate):
- Data Collection: 30-60 min
- Index Building: 10-20 min
- Query Processing: 1-3 sec
- Full Evaluation: 30-60 min
