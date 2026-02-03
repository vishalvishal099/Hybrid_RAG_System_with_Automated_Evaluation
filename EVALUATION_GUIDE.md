# Comprehensive Evaluation System Implementation

## Overview

This implements a complete evaluation framework for the Hybrid RAG System with all required components:

### ✅ Implemented Features

#### 1. **Semantic Chunking (200-400 tokens, 50-token overlap)**
- **File**: `src/semantic_chunker.py`
- **Features**:
  - Paragraph-aware splitting
  - Anchor preservation (section headers)
  - Deduplication using content hashing
  - Token counting with tiktoken
  - Proper overlapping windows

#### 2. **RRF Fusion (k=60)**
- **File**: `src/rrf_fusion.py`
- **Formula**: `RRF_score(d) = Σ 1/(k + rank_i(d))`
- **Combines**: Dense (FAISS) + Sparse (BM25) results

#### 3. **Comprehensive Metrics**
- **File**: `evaluation/comprehensive_metrics.py`
- **Metrics Implemented**:
  - **MRR** (Mean Reciprocal Rank) - URL-based retrieval
  - **Recall@10** - URL-based retrieval coverage
  - **Answer F1** - Token overlap between generated and ground truth
  - **Precision/Recall** - Answer quality components
  - **Exact Match** - Binary correctness score

#### 4. **Evaluation Dataset**
- **File**: `evaluation/create_dataset.py`
- **Dataset**: `data/evaluation_dataset.json`
- **Contains**: 10 test cases with:
  - Questions
  - Ground truth URLs
  - Expected answers

#### 5. **Ablation Studies**
- **Dense-only** (FAISS only)
- **Sparse-only** (BM25 only)
- **Hybrid RRF** (k=60 fusion)

#### 6. **LLM-as-Judge**
- **Dimensions evaluated**:
  - Relevance (query-answer alignment)
  - Correctness (factual accuracy)
  - Completeness (comprehensive coverage)
  - Overall score

#### 7. **Complete Evaluation Pipeline**
- **File**: `evaluation/run_evaluation.py`
- **Outputs**:
  - JSON results
  - Visualizations (PNG charts)
  - Text report

---

## Usage Guide

### Step 1: Update RAG System with RRF Fusion

Add these methods to `src/rag_system.py`:

```python
from src.rrf_fusion import RRFFusion

class HybridRAGSystem:
    def __init__(self, ...):
        # ... existing code ...
        self.rrf = RRFFusion(k=60)
    
    def retrieve_with_rrf(self, query: str, top_k: int = 10):
        """Retrieve using RRF fusion"""
        # Get FAISS results
        query_embedding = self.embedding_model.encode([query])
        faiss_scores, faiss_indices = self.faiss_index.search(query_embedding, top_k * 2)
        
        # Get BM25 results
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Apply RRF fusion
        fused = self.rrf.fuse_rankings(
            list(zip(faiss_indices[0], faiss_scores[0])),
            list(zip(bm25_indices, bm25_scores[bm25_indices]))
        )
        
        # Return top-K results
        results = []
        for doc_id, score in fused[:top_k]:
            if doc_id < len(self.corpus):
                chunk = self.corpus[doc_id]
                results.append({
                    'text': chunk['text'],
                    'title': chunk.get('title', 'Unknown'),
                    'url': chunk.get('url'),
                    'score': score
                })
        return results
    
    def dense_search_only(self, query: str, top_k: int = 10):
        """FAISS-only search for ablation"""
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.corpus):
                chunk = self.corpus[idx]
                results.append({
                    'text': chunk['text'],
                    'title': chunk.get('title'),
                    'url': chunk.get('url'),
                    'score': score
                })
        return results
    
    def sparse_search_only(self, query: str, top_k: int = 10):
        """BM25-only search for ablation"""
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.corpus):
                chunk = self.corpus[idx]
                results.append({
                    'text': chunk['text'],
                    'title': chunk.get('title'),
                    'url': chunk.get('url'),
                    'score': bm25_scores[idx]
                })
        return results
```

### Step 2: Run Complete Evaluation

```python
from src.rag_system import HybridRAGSystem
from evaluation.run_evaluation import ComprehensiveEvaluationPipeline
from evaluation.create_dataset import load_evaluation_dataset

# Load your RAG system
rag_system = HybridRAGSystem(config_path="config.yaml")
rag_system.load_corpus()
rag_system.load_indexes()

# Load test cases
test_cases = load_evaluation_dataset()

# Create and run pipeline
pipeline = ComprehensiveEvaluationPipeline(rag_system)
results = pipeline.run_complete_evaluation(test_cases)

# Results saved in evaluation/results/
```

### Step 3: Rebuild Corpus with Semantic Chunking (Optional)

To use the improved semantic chunking:

```python
from src.semantic_chunker import process_corpus_with_semantic_chunking
import json

# Load raw Wikipedia data
with open('data/raw_wikipedia_data.json', 'r') as f:
    raw_docs = json.load(f)

# Process with semantic chunking (200-400 tokens, 50 overlap)
corpus = process_corpus_with_semantic_chunking(
    corpus_data=raw_docs,
    min_tokens=200,
    max_tokens=400,
    overlap_tokens=50
)

# Save new corpus
with open('data/corpus_semantic.json', 'w') as f:
    json.dump(corpus, f, indent=2)

# Rebuild indexes using src/indexing.py with new corpus
```

---

## Evaluation Metrics Explained

### 1. **MRR (Mean Reciprocal Rank)**
- **Purpose**: Measures how quickly the correct document appears in results
- **Formula**: `MRR = 1/rank` (rank of first relevant document)
- **Best**: 1.0 (correct doc is rank 1)
- **Interpretation**: MRR=0.5 means correct doc at rank 2 on average

### 2. **Recall@10**
- **Purpose**: Measures coverage of relevant documents in top-10
- **Formula**: `Recall = |retrieved ∩ relevant| / |relevant|`
- **Best**: 1.0 (all relevant docs in top-10)

### 3. **Answer F1**
- **Purpose**: Token-level overlap between generated and ground truth
- **Formula**: `F1 = 2 * (P * R) / (P + R)`
- **Components**:
  - Precision: % of generated tokens that are correct
  - Recall: % of ground truth tokens that appear in generation

### 4. **LLM-as-Judge**
- **Relevance**: Does answer address the question?
- **Correctness**: Are the facts accurate?
- **Completeness**: Is all important info included?

---

## Expected Output

### Console Output
```
==================================================================
STARTING COMPREHENSIVE EVALUATION PIPELINE
==================================================================
Test cases: 10

==================================================================
RETRIEVAL EVALUATION
==================================================================
  Evaluated 10/10 queries...

✓ Retrieval Evaluation Complete:
  - Mean MRR: 0.7250
  - Mean Recall@10: 0.8500

==================================================================
GENERATION EVALUATION
==================================================================
  Evaluated 10 answers...

✓ Generation Evaluation Complete:
  - Mean Answer F1: 0.5432
  - Mean Precision: 0.6123
  - Mean Recall: 0.4891

==================================================================
ABLATION STUDY
==================================================================
  Evaluated 10 queries...

✓ Ablation Study Complete:
  - Dense-only MRR: 0.6500
  - Sparse-only MRR: 0.5800
  - Hybrid RRF MRR: 0.7250

==================================================================
LLM-AS-JUDGE EVALUATION
==================================================================
  Evaluated 10/10 samples...

✓ LLM Judge Evaluation Complete:
  - Relevance: 0.7234
  - Correctness: 0.6891
  - Completeness: 0.6234
  - Overall: 0.6786
```

### Generated Files

1. **`evaluation/results/evaluation_results.json`**
   - Complete JSON with all metrics

2. **`evaluation/results/ablation_study.png`**
   - Bar chart comparing Dense/Sparse/Hybrid

3. **`evaluation/results/metrics_overview.png`**
   - Three-panel chart: Retrieval, Generation, LLM Judge

4. **`evaluation/results/evaluation_report.txt`**
   - Text summary of all results

---

## Key Implementation Details

### RRF Fusion Algorithm
```python
def rrf_score(doc_id, rankings, k=60):
    score = 0
    for ranking in rankings:
        if doc_id in ranking:
            rank = ranking.index(doc_id) + 1
            score += 1 / (k + rank)
    return score
```

### Semantic Chunking Flow
1. **Extract anchors** (section headers)
2. **Split into paragraphs** (preserve structure)
3. **Tokenize sentences** (NLTK)
4. **Build overlapping chunks** (200-400 tokens)
5. **Deduplicate** (MD5 hash comparison)
6. **Add metadata** (doc_id, position, anchor, URL)

### Evaluation Pipeline Flow
```
Test Cases → Retrieval → MRR/Recall@10
          ↓
      Generation → Answer F1
          ↓
   Ablation Study → Dense/Sparse/Hybrid comparison
          ↓
    LLM Judge → Relevance/Correctness/Completeness
          ↓
      Generate Report + Visualizations
```

---

## Next Steps

1. **Integrate RRF into main system**: Update `src/rag_system.py`
2. **Run evaluation**: Execute pipeline on your system
3. **Analyze results**: Check which method performs best
4. **Optional**: Rebuild corpus with semantic chunking
5. **Document findings**: Use results in your assignment report

---

## Files Created

- ✅ `src/semantic_chunker.py` - Semantic chunking implementation
- ✅ `src/rrf_fusion.py` - RRF fusion algorithm
- ✅ `evaluation/comprehensive_metrics.py` - All evaluation metrics
- ✅ `evaluation/create_dataset.py` - Ground truth dataset
- ✅ `evaluation/run_evaluation.py` - Complete pipeline
- ✅ `data/evaluation_dataset.json` - 10 test cases with ground truth

---

## Troubleshooting

**Issue**: Import errors for semantic_chunker
**Fix**: Ensure tiktoken and nltk are installed:
```bash
pip install tiktoken nltk
```

**Issue**: "No module named 'src'"
**Fix**: Run from project root directory

**Issue**: Missing corpus URLs
**Fix**: Ensure your corpus includes 'url' field for each chunk

---

## Configuration

All parameters match requirements:
- ✅ Chunk size: 200-400 tokens
- ✅ Overlap: 50 tokens
- ✅ RRF k: 60
- ✅ Top-K: 10 for retrieval
- ✅ Top-N: 5 for generation context

No changes needed!
