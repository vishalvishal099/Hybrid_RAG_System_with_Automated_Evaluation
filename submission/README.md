# 📦 Hybrid RAG System with Automated Evaluation - Submission Package

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

**Student Submission for Conversational AI Assignment 2**

**GitHub Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

---

## 📁 Folder Structure

```
submission/
├── README.md                    # This file
├── 01_source_code/              # All Python source files
├── 02_data/                     # Dataset files (corpus, questions, URLs)
├── 03_vector_database/          # ChromaDB vector database
├── 04_evaluation_results/       # CSV, JSON, HTML evaluation outputs
├── 05_reports/                  # PDF report and evaluation documents
├── 06_documentation/            # Architecture, metrics, error analysis
├── 07_visualizations/           # Charts and diagrams
└── 08_screenshots/              # UI screenshots
```

---

## 📂 01_source_code/

Core Python implementation files:

| File | Description | GitHub URL |
|------|-------------|------------|
| `chromadb_rag_system.py` | Main RAG system with Dense, Sparse, Hybrid retrieval | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/chromadb_rag_system.py) |
| `app_chromadb.py` | Streamlit UI application | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/app_chromadb.py) |
| `evaluate_chromadb_fast.py` | Automated evaluation script | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluate_chromadb_fast.py) |
| `generate_report.py` | HTML/visualization report generator | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/generate_report.py) |
| `api_chromadb.py` | REST API implementation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/api_chromadb.py) |
| `requirements.txt` | Python dependencies | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/requirements.txt) |
| `start_ui.sh` | UI startup script | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/start_ui.sh) |

---

## 📂 02_data/

Dataset files for the RAG system:

| File | Size | Description |
|------|------|-------------|
| `corpus.json` | 14.5 MB | Preprocessed Wikipedia articles (501 articles, 7519 chunks) |
| `questions_100.json` | 50 KB | 100 evaluation questions with ground truth answers |
| `fixed_urls.json` | 10 KB | 200 fixed Wikipedia URLs |

### Question Distribution:
- Factual: 59 questions
- Comparative: 15 questions
- Inferential: 11 questions
- Multi-hop: 15 questions

---

## 📂 03_vector_database/

ChromaDB vector database:

| File | Size | Description |
|------|------|-------------|
| `chroma_db/chroma.sqlite3` | 212 MB | SQLite database with 7519 vectors |
| `chroma_db/bm25_index.pkl` | 12 MB | BM25 sparse retrieval index |
| `chroma_db/bm25_corpus.pkl` | 12 MB | BM25 corpus data |

---

## 📂 04_evaluation_results/

Evaluation outputs (100 questions × 3 methods = 300 results):

| File | Size | Description |
|------|------|-------------|
| `evaluation_results_chromadb.csv` | 213 KB | Detailed results for all 300 evaluations |
| `evaluation_summary_chromadb.json` | 1 KB | Summary statistics |
| `evaluation_report_chromadb.html` | 10 KB | Interactive HTML report |

### Evaluation Metrics Summary:

| Method | MRR | Recall@10 | Avg Answer F1 |
|--------|-----|-----------|---------------|
| Dense | 0.3025 | 0.33 | ~0.05 |
| **Sparse (BM25)** | **0.4392** | **0.47** | ~0.05 |
| Hybrid (RRF) | 0.3783 | 0.43 | ~0.05 |

---

## 📂 05_reports/

Comprehensive evaluation reports:

| File | Size | Description |
|------|------|-------------|
| `Hybrid_RAG_Evaluation_Report.pdf` | 16 KB | **Main PDF submission report** |
| `EVALUATION_REPORT.md` | 8 KB | Markdown version of report |
| `evaluation_report.tex` | 8 KB | LaTeX source for report |

---

## 📂 06_documentation/

Technical documentation:

| File | Description |
|------|-------------|
| `METRIC_JUSTIFICATION.md` | **Detailed justification for MRR, Recall@10, Answer F1 metrics** |
| `ERROR_ANALYSIS.md` | **Failure categorization and pattern analysis** |
| `architecture_diagram.png` | System architecture visualization |
| `data_flow_diagram.png` | Data flow through the RAG pipeline |
| `README.md` | Main project README |
| `SUBMISSION_README.md` | Submission guide |

---

## 📂 07_visualizations/

Charts and visualizations:

| File | Description |
|------|-------------|
| `comparison_metrics.png` | Bar chart comparing Dense/Sparse/Hybrid metrics |
| `distribution_charts.png` | Score distribution histograms |
| `performance_metrics.png` | Response time analysis |
| `error_analysis_charts.png` | Failure category breakdown |
| `retrieval_heatmap.png` | Chunk relevance heatmap |

---

## 📂 08_screenshots/

UI screenshots demonstrating system functionality:

| File | Description |
|------|-------------|
| `01_query_interface.png` | Main query interface with input and results |
| `02_method_comparison.png` | Comparison of retrieval methods |
| `03_evaluation_results.png` | Evaluation metrics display |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd 01_source_code/
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Copy data and database to working directory first
streamlit run app_chromadb.py
```

### 3. Run Evaluation
```bash
python evaluate_chromadb_fast.py
```

---

## 📊 Key Results

### Retrieval Performance (100 questions each method)

| Metric | Dense | Sparse (BM25) | Hybrid (RRF) |
|--------|-------|---------------|--------------|
| **MRR** | 0.3025 | **0.4392** | 0.3783 |
| **Recall@10** | 0.33 | **0.47** | 0.43 |
| **Answer F1** | ~0.05 | ~0.05 | ~0.05 |

### Key Findings:
1. **BM25 (Sparse) outperforms** both Dense and Hybrid on this Wikipedia dataset
2. **Hybrid (RRF)** provides balanced performance between Dense and Sparse
3. **Low Answer F1** indicates opportunity for LLM improvement

---

## 📝 Requirements Completed

| Section | Status | Evidence |
|---------|--------|----------|
| Section 1: Hybrid RAG System | ✅ 100% | `chromadb_rag_system.py`, `app_chromadb.py` |
| Section 2.1: Question Generation | ✅ 100% | `questions_100.json` |
| Section 2.2.1: MRR Metric | ✅ 100% | Evaluation results |
| Section 2.2.2: Custom Metrics | ✅ 100% | `METRIC_JUSTIFICATION.md` |
| Section 2.4: Automated Pipeline | ✅ 100% | Evaluation scripts |
| Section 2.5: Report Contents | ✅ 95% | PDF, visualizations, analysis |

---

## 🔗 Links

- **GitHub Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

---

**Submission Date:** February 7, 2026
