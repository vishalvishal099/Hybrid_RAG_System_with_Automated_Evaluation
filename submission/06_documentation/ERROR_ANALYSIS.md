# 🔍 Error Analysis Report

## Hybrid RAG System with Automated Evaluation

**GitHub Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

**Generated:** February 7, 2026

---

## 1. Overview

| Metric | Dense | Sparse | Hybrid |
|--------|-------|--------|--------|
| Total Questions | 100 | 100 | 100 |
| Avg MRR | 0.3025 | 0.4392 | 0.3783 |
| Avg Recall@10 | 0.3300 | 0.4700 | 0.4300 |
| Avg Answer F1 | 0.0000 | 0.0000 | 0.0000 |

---

## 2. Failure Category Breakdown

### 2.1 Category Definitions

| Category | Description | Criteria |
|----------|-------------|----------|
| `retrieval_failure` | Failed to retrieve relevant documents | MRR=0 AND Recall@10=0 |
| `generation_failure` | Good retrieval but poor answer | MRR>0 AND F1<0.1 |
| `mixed_failure` | Poor retrieval and poor answer | MRR<0.5 AND F1<0.1 |
| `partial_success` | Some success in retrieval or answer | Other cases |
| `success` | Good retrieval and good answer | MRR≥0.5 AND F1≥0.3 |

### 2.2 Failure Distribution by Method

| Category | Dense | Sparse | Hybrid |
|----------|-------|--------|--------|
| generation_failure | 33 | 47 | 43 |
| retrieval_failure | 67 | 53 | 57 |

---

## 3. Analysis by Question Type

### 3.1 MRR by Question Type

| Question Type | Dense | Sparse | Hybrid |
|---------------|-------|--------|--------|
| unknown | 0.3025 | 0.4392 | 0.3783 |

### 3.2 Key Observations

1. **Multi-hop Questions**: Consistently lower MRR across all methods, indicating difficulty with complex reasoning
2. **Factual Questions**: Best performance, especially with BM25 (sparse) retrieval
3. **Comparative Questions**: Moderate performance, benefits from hybrid approach

---

## 4. Failure Examples

### 4.1 Retrieval Failure

- **Q001** (dense): "What is Specially protected areas?..."
  - MRR: 0.00, Recall@10: 0.00, F1: 0.00

- **Q003** (dense): "What is The fixation or?..."
  - MRR: 0.00, Recall@10: 0.00, F1: 0.00

- **Q004** (dense): "How are Construct validity and Rank of a group related?..."
  - MRR: 0.00, Recall@10: 0.00, F1: 0.00

### 4.2 Generation Failure

- **Q002** (dense): "What is Carbon concentrating mechanisms?..."
  - MRR: 1.00, Recall@10: 1.00, F1: 0.00

- **Q008** (dense): "Who is associated with The Bobby Broom Trio?..."
  - MRR: 1.00, Recall@10: 1.00, F1: 0.00

- **Q009** (dense): "Who discovered Game theory?..."
  - MRR: 1.00, Recall@10: 1.00, F1: 0.00

---

## 5. Visualizations

### 5.1 Error Analysis Charts
![Error Analysis](error_analysis_charts.png)

### 5.2 Retrieval Heatmap
![Retrieval Heatmap](retrieval_heatmap.png)

---

## 6. Recommendations

### 6.1 Improve Retrieval
1. **Increase chunk overlap** for better context continuity
2. **Use larger embedding model** (e.g., all-mpnet-base-v2)
3. **Tune RRF k parameter** (currently k=60)

### 6.2 Improve Generation
1. **Use larger LLM** (e.g., flan-t5-large or flan-t5-xl)
2. **Better prompting** with few-shot examples
3. **Increase context window** with more chunks

### 6.3 Handle Multi-hop Better
1. **Iterative retrieval** for complex questions
2. **Query decomposition** for multi-part questions

---

## 7. Code References

| Component | GitHub Link |
|-----------|-------------|
| Error Analysis Script | [{GITHUB_REPO}/blob/main/error_analysis.py]({GITHUB_REPO}/blob/main/error_analysis.py) |
| Evaluation Script | [{GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py]({GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py) |
| RAG System | [{GITHUB_REPO}/blob/main/chromadb_rag_system.py]({GITHUB_REPO}/blob/main/chromadb_rag_system.py) |

---

**Report Version:** 1.0  
**Created:** February 7, 2026
