# Hybrid RAG System with Automated Evaluation

## 📋 Submission Package

**GitHub Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

**Course:** BITS Pilani - Conversational AI Assignment 2

**Date:** February 7, 2026

---

## 📁 Folder Structure

```
Hybrid_RAG_System_with_Automated_Evaluation/
├── src/                           # Source code
│   ├── chromadb_rag_system.py     # Main RAG system implementation
│   ├── app_chromadb.py            # Streamlit UI
│   ├── evaluate_chromadb_fast.py  # Evaluation pipeline
│   └── generate_report.py         # Report generation
│
├── data/                          # Data files
│   ├── fixed_urls.json            # 200 fixed Wikipedia URLs
│   ├── corpus.json                # Preprocessed corpus (14.5MB)
│   ├── questions_100.json         # 100 evaluation questions
│   └── indexes/                   # BM25 index files
│
├── chroma_db/                     # Vector database (212MB)
│
├── docs/                          # Documentation
│   ├── METRIC_JUSTIFICATION.md    # Metric selection rationale
│   ├── ERROR_ANALYSIS.md          # Failure analysis
│   ├── EVALUATION_REPORT.md       # Full evaluation report
│   ├── architecture_diagram.png   # System architecture
│   └── *.png                      # Visualizations
│
├── reports/                       # Generated reports
│   └── Hybrid_RAG_Evaluation_Report.pdf
│
├── screenshots/                   # UI screenshots
│   ├── 01_query_interface.png
│   ├── 02_method_comparison.png
│   └── 03_evaluation_results.png
│
├── evaluation/                    # Evaluation results
│   ├── evaluation_results_chromadb.csv
│   ├── evaluation_summary_chromadb.json
│   └── evaluation_report_chromadb.html
│
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
└── start_ui.sh                    # Quick start script
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB+ RAM (for embeddings and LLM)

### Installation

```bash
# Clone repository
git clone https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation.git
cd Hybrid_RAG_System_with_Automated_Evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

## 🏗️ System Architecture

### Components

| Component | Technology | Description |
|-----------|------------|-------------|
| Dense Retrieval | ChromaDB + all-MiniLM-L6-v2 | 384-dim vector embeddings |
| Sparse Retrieval | BM25 + NLTK | Keyword-based matching |
| Fusion | RRF (k=60) | Reciprocal Rank Fusion |
| Generation | FLAN-T5-base | Text-to-text transformer |
| Interface | Streamlit | Interactive web UI |

### Data Flow

1. **Query Input** → User enters question via Streamlit
2. **Dense Search** → Embed query, search ChromaDB
3. **Sparse Search** → BM25 keyword matching
4. **RRF Fusion** → Combine rankings with k=60
5. **Generation** → FLAN-T5 generates answer from context
6. **Display** → Show answer, sources, and metrics

---

## 📊 Evaluation Results

### Performance Summary

| Method | MRR | Recall@10 | Avg Time |
|--------|-----|-----------|----------|
| Dense | 0.3025 | 0.33 | 5.86s |
| **Sparse (BM25)** | **0.4392** | **0.47** | 5.53s |
| Hybrid (RRF) | 0.3783 | 0.43 | 6.37s |

**Key Finding:** BM25 outperforms Dense by 45% on MRR for Wikipedia-based QA.

### Evaluation Dataset

- **100 Questions** across 4 types:
  - Factual (59)
  - Comparative (15)
  - Inferential (11)
  - Multi-hop (15)

---

## 📚 Documentation

| Document | Description | Link |
|----------|-------------|------|
| Metric Justification | Why MRR, Recall@10, Answer F1 | [docs/METRIC_JUSTIFICATION.md](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/METRIC_JUSTIFICATION.md) |
| Error Analysis | Failure categorization | [docs/ERROR_ANALYSIS.md](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/ERROR_ANALYSIS.md) |
| Full Report | Comprehensive evaluation | [docs/EVALUATION_REPORT.md](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/EVALUATION_REPORT.md) |
| PDF Report | Printable report | [reports/Hybrid_RAG_Evaluation_Report.pdf](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/reports/Hybrid_RAG_Evaluation_Report.pdf) |

---

## 📸 Screenshots

### Query Interface
![Query Interface](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/01_query_interface.png)

### Method Comparison
![Method Comparison](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/02_method_comparison.png)

### Evaluation Results
![Evaluation Results](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/03_evaluation_results.png)

---

## 🔗 Key Files

### Source Code
- [chromadb_rag_system.py](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/chromadb_rag_system.py) - Core RAG implementation
- [app_chromadb.py](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/app_chromadb.py) - Streamlit UI
- [evaluate_chromadb_fast.py](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluate_chromadb_fast.py) - Evaluation pipeline

### Data Files
- [data/questions_100.json](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/questions_100.json) - 100 evaluation questions
- [data/fixed_urls.json](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/fixed_urls.json) - 200 fixed Wikipedia URLs

### Results
- [evaluation_results_chromadb.csv](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_results_chromadb.csv) - 300 evaluation rows
- [evaluation_summary_chromadb.json](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_summary_chromadb.json) - Summary metrics

---

## 📋 Requirements Checklist

### Section 1: Hybrid RAG System (10 pts) ✅
- [x] Dense Vector Retrieval (ChromaDB)
- [x] Sparse Keyword Retrieval (BM25)
- [x] RRF Fusion (k=60)
- [x] Response Generation (FLAN-T5)
- [x] Interactive UI (Streamlit)

### Section 2: Evaluation Framework (10 pts)
- [x] 100 Q&A pairs generated
- [x] MRR metric implemented
- [x] Recall@10 metric implemented
- [x] Answer F1 metric implemented
- [x] Automated evaluation pipeline
- [x] HTML/CSV/JSON reports

### Submission Requirements
- [x] Python source code
- [x] PDF evaluation report
- [x] Screenshots (3+)
- [x] README documentation
- [x] 100-question dataset
- [x] Evaluation results

---

## 📄 License

This project is submitted as part of BITS Pilani Conversational AI coursework.

---

**Generated:** February 7, 2026
