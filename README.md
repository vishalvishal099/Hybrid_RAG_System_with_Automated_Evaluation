# Hybrid RAG System with Automated Evaluation

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

A comprehensive implementation of a **Hybrid RAG System** combining **Dense Vector Retrieval (ChromaDB)**, **Sparse Keyword Retrieval (BM25)**, and **Reciprocal Rank Fusion (RRF)** to answer questions from Wikipedia articles.

**GitHub Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation.git
cd Hybrid_RAG_System_with_Automated_Evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Start Streamlit UI
./start_ui.sh

# Or manually:
streamlit run app_chromadb.py
```

### Run Evaluation

```bash
# Full evaluation (100 questions × 3 methods)
python evaluate_chromadb_fast.py

# Generate reports
python generate_report.py
```

---

## 🎯 Project Overview

This project implements a state-of-the-art Hybrid RAG system that:
- Combines **dense** (ChromaDB + MiniLM) and **sparse** (BM25) retrieval
- Uses **Reciprocal Rank Fusion (RRF)** with k=60 to merge results
- Generates answers using **FLAN-T5** language model
- Includes comprehensive evaluation with **100 generated questions**
- Features automated evaluation pipeline with MRR, Recall@10, and Answer F1

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└──────────────────────┬──────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
    ┌───────▼────────┐   ┌────────▼───────┐
    │ Dense Retrieval│   │Sparse Retrieval│
    │ (ChromaDB +    │   │    (BM25 +     │
    │  MiniLM-L6-v2) │   │     NLTK)      │
    └───────┬────────┘   └────────┬───────┘
            │                     │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │ Reciprocal Rank     │
            │ Fusion (k=60)       │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   Top-K Chunks      │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Answer Generation  │
            │  (FLAN-T5-base)     │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   Generated Answer  │
            │   + Source URLs     │
            └─────────────────────┘
```

---

## 🗂️ Project Structure

```
Hybrid_RAG_System_with_Automated_Evaluation/
│
├── chromadb_rag_system.py      # Core RAG implementation
├── app_chromadb.py             # Streamlit UI (244 lines)
├── evaluate_chromadb_fast.py   # Evaluation pipeline
├── generate_report.py          # Report generator
├── start_ui.sh                 # Quick start script
│
├── data/
│   ├── fixed_urls.json         # 200 fixed Wikipedia URLs
│   ├── corpus.json             # Preprocessed corpus (14.5MB)
│   ├── questions_100.json      # 100 evaluation questions
│   └── indexes/                # BM25 index files
│
├── chroma_db/                  # ChromaDB vector database (212MB)
│
├── docs/
│   ├── METRIC_JUSTIFICATION.md # Metric selection rationale
│   ├── ERROR_ANALYSIS.md       # Failure analysis
│   ├── EVALUATION_REPORT.md    # Full evaluation report
│   ├── architecture_diagram.png
│   └── *.png                   # Visualizations
│
├── reports/
│   └── Hybrid_RAG_Evaluation_Report.pdf
│
├── screenshots/
│   ├── 01_query_interface.png
│   ├── 02_method_comparison.png
│   └── 03_evaluation_results.png
│
├── evaluation_results_chromadb.csv     # 300 evaluation rows
├── evaluation_summary_chromadb.json    # Summary metrics
├── evaluation_report_chromadb.html     # HTML report
│
└── README.md                   # This file
```

---

## 📈 Evaluation Results

### Performance Summary

| Method | MRR | Recall@10 | Avg Time (s) | Questions |
|--------|-----|-----------|--------------|-----------|
| Dense (ChromaDB) | 0.3025 | 0.33 | 5.86 | 100 |
| **Sparse (BM25)** | **0.4392** | **0.47** | 5.53 | 100 |
| Hybrid (RRF) | 0.3783 | 0.43 | 6.37 | 100 |

**Key Finding:** BM25 (Sparse) outperforms Dense retrieval by **45%** on MRR for Wikipedia-based QA.

### Question Distribution

| Type | Count | Description |
|------|-------|-------------|
| Factual | 59 | Direct fact-based questions |
| Comparative | 15 | Questions comparing concepts |
| Inferential | 11 | Reasoning-based questions |
| Multi-hop | 15 | Questions requiring multiple sources |
| **Total** | **100** | - |

---

## 📚 Documentation

| Document | Description | Link |
|----------|-------------|------|
| Metric Justification | Why MRR, Recall@10, Answer F1 | [docs/METRIC_JUSTIFICATION.md](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/METRIC_JUSTIFICATION.md) |
| Error Analysis | Failure categorization | [docs/ERROR_ANALYSIS.md](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/ERROR_ANALYSIS.md) |
| Full Report | Comprehensive evaluation | [docs/EVALUATION_REPORT.md](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/EVALUATION_REPORT.md) |
| PDF Report | Printable report | [reports/Hybrid_RAG_Evaluation_Report.pdf](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/reports/Hybrid_RAG_Evaluation_Report.pdf) |

---

## 🔗 Key Source Files

| File | Purpose | Link |
|------|---------|------|
| `chromadb_rag_system.py` | Core RAG implementation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/chromadb_rag_system.py) |
| `app_chromadb.py` | Streamlit UI | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/app_chromadb.py) |
| `evaluate_chromadb_fast.py` | Evaluation pipeline | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluate_chromadb_fast.py) |
| `generate_report.py` | Report generation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/generate_report.py) |

---

## 📸 Screenshots

### Query Interface
![Query Interface](screenshots/01_query_interface.png)

### Method Comparison  
![Method Comparison](screenshots/02_method_comparison.png)

### Evaluation Results
![Evaluation Results](screenshots/03_evaluation_results.png)

---

## 🛠️ Technical Details

### Components

| Component | Technology | Details |
|-----------|------------|---------|
| Dense Retrieval | ChromaDB + all-MiniLM-L6-v2 | 384-dim embeddings, 7,519 chunks |
| Sparse Retrieval | BM25 + NLTK | Tokenization, stopwords, stemming |
| Fusion | RRF | Reciprocal Rank Fusion with k=60 |
| Generation | FLAN-T5-base | 248M parameter text-to-text model |
| UI | Streamlit | Interactive web interface |
| Database | ChromaDB | Persistent SQLite backend (212MB) |

### Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MRR** | (1/Q) × Σ(1/rank_i) | Measures retrieval quality |
| **Recall@10** | \|Relevant ∩ Retrieved@10\| / \|Relevant\| | Coverage in top 10 |
| **Answer F1** | 2×(P×R)/(P+R) | Token overlap with ground truth |

---

## 📋 Requirements Checklist

### ✅ Section 1: Hybrid RAG System (10 pts)
- [x] Dense Vector Retrieval (ChromaDB + MiniLM)
- [x] Sparse Keyword Retrieval (BM25)
- [x] RRF Fusion (k=60)
- [x] Response Generation (FLAN-T5)
- [x] Interactive UI (Streamlit)

### ✅ Section 2: Evaluation Framework (10 pts)
- [x] 100 Q&A pairs generated
- [x] MRR metric implemented
- [x] Recall@10 metric implemented
- [x] Answer F1 metric implemented
- [x] Automated evaluation pipeline
- [x] HTML/CSV/JSON/PDF reports

### ✅ Submission Requirements
- [x] Python source code (24 files)
- [x] PDF evaluation report
- [x] Screenshots (3+)
- [x] README documentation
- [x] 100-question dataset
- [x] Evaluation results (300 rows)

---

## 📄 License

This project is submitted as part of BITS Pilani Conversational AI coursework.

---

**Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

**Last Updated:** February 7, 2026
