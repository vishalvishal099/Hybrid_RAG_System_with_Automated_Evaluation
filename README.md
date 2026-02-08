# Hybrid RAG System with Automated Evaluation

A comprehensive Hybrid RAG System combining Dense Vector Retrieval, Sparse Keyword Retrieval, and Reciprocal Rank Fusion for Wikipedia-based Question Answering.

---

## Quick Start

**Prerequisites:** Python 3.10+, 4GB+ RAM

```bash
# Clone and setup
git clone https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation.git
cd Hybrid_RAG_System_with_Automated_Evaluation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run application
streamlit run app_chromadb.py --server.port 8502

# Run evaluation
python evaluate_chromadb_fast.py
```

**Access:** http://localhost:8502

---

## Project Overview

| Component | Technology |
|-----------|------------|
| Dense Retrieval | ChromaDB + MiniLM-L6-v2 (384 dims) |
| Sparse Retrieval | BM25 + NLTK |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Generation | FLAN-T5-Base (248M params) |
| Interface | Streamlit Dashboard |

**Dataset:** 500 Wikipedia URLs, 7,519 chunks, 100 evaluation questions

---

## System Architecture

```
                      User Query
                          │
                          ▼
                   Query Processing
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     Dense Retrieval            Sparse Retrieval
       (ChromaDB)                    (BM25)
            │                           │
            └─────────────┬─────────────┘
                          ▼
                   RRF Fusion (k=60)
                          │
                          ▼
                    Top-K Chunks
                          │
                          ▼
                  Answer Generation
                    (FLAN-T5)
                          │
                          ▼
                  Generated Answer
```

---

## Evaluation Results

| Method | MRR | Recall@10 | Time |
|--------|-----|-----------|------|
| Dense | 0.3025 | 0.33 | 5.86s |
| **Sparse (BM25)** | **0.4392** | **0.47** | **5.53s** |
| Hybrid (RRF) | 0.3783 | 0.43 | 6.37s |

**Key Finding:** BM25 outperforms Dense by 45% on MRR.

---

## Project Structure

```
├── chromadb_rag_system.py    # Core RAG
├── app_chromadb.py           # Streamlit UI
├── evaluate_chromadb_fast.py # Evaluation
├── error_analysis.py         # Error analysis
├── data/
│   ├── fixed_urls.json       # 200 URLs
│   ├── corpus.json           # 7,519 chunks
│   └── questions_100.json    # 100 Q&A pairs
├── evaluation/
│   ├── metrics.py            # MRR, Recall, F1
│   ├── novel_metrics.py      # Custom metrics
│   └── pipeline.py           # Eval pipeline
├── docs/
│   ├── METRIC_JUSTIFICATION.md
│   └── *.png                 # Diagrams
├── screenshots/              # UI screenshots
└── submission/               # Submission package
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [SUBMISSION_DELIVERABLES.md](SUBMISSION_DELIVERABLES.md) | Assignment mapping |
| [QUICK_ACCESS_LINKS.md](QUICK_ACCESS_LINKS.md) | All file links |
| [docs/METRIC_JUSTIFICATION.md](docs/METRIC_JUSTIFICATION.md) | Metric rationale |

---

## Requirements Checklist

**Part 1: Hybrid RAG System**
- [x] Dense Vector Retrieval (ChromaDB)
- [x] Sparse Keyword Retrieval (BM25)
- [x] RRF Fusion (k=60)
- [x] FLAN-T5 Generation
- [x] Streamlit Dashboard

**Part 2: Evaluation Framework**
- [x] 100 Q&A pairs
- [x] MRR, Recall@10, Answer F1
- [x] Automated pipeline
- [x] PDF/CSV/JSON reports

---

## Contributors

| Name | BITS ID |
|------|---------|
| VISHAL SINGH | 2024AA05641 |
| GOBIND SAH | 2024AA05643 |
| YASH VERMA | 2024AA05640 |
| AVISHI GUPTA | 2024AA05055 |
| SAYAN MANNA | 2024AB05304 |

---

**BITS Pilani - Conversational AI Assignment**

**Repository:** https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation
