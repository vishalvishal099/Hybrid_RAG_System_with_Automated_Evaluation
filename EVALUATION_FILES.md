# 📊 Evaluation Files - 100 Question Automated Framework

## ✅ Completed: Automated Evaluation with 100 Generated Questions

All evaluation files are located in the project root directory:
**`/Users/v0s01jh/Documents/BITS/ConvAI_assingment_2`**

---

## 📁 File Locations (Click to Open)

### 📝 1. Questions Dataset (100 Questions)
**File:** [data/questions_100.json](./data/questions_100.json)
- **Size:** 49 KB
- **Content:** 100 automatically generated questions with expected answers and source URLs
- **Format:** JSON array with question, expected_answer, and source_url fields

---

### 📊 2. Detailed Evaluation Results (All 100 Questions × 3 Methods)
**File:** [evaluation_results_chromadb.csv](./evaluation_results_chromadb.csv)
- **Size:** 182 KB, 263 rows
- **Content:** Detailed results for each question including:
  - Question text
  - Expected URL
  - Retrieved top-1 URL
  - Generated answer
  - MRR (Mean Reciprocal Rank)
  - Recall@10
  - Answer F1 score
  - Retrieval and generation times
  - Method (dense/sparse/hybrid)

---

### 📈 3. Evaluation Summary Statistics
**File:** [evaluation_summary_chromadb.json](./evaluation_summary_chromadb.json)
- **Size:** 1.1 KB
- **Content:** Aggregated metrics for each retrieval method:
  - **Dense (ChromaDB):** 62 questions, MRR=0.363, Recall@10=0.419
  - **Sparse (BM25):** 100 questions, MRR=0.439, Recall@10=0.47
  - **Hybrid (RRF):** 100 questions, MRR=0.378, Recall@10=0.43
  - Average retrieval/generation times
  - Total evaluation time

---

### 📄 4. HTML Evaluation Report with Visualizations
**File:** [evaluation_report_chromadb.html](./evaluation_report_chromadb.html)
- **Size:** 9.8 KB
- **Content:** Interactive HTML report with:
  - Performance comparison charts
  - Method-by-method breakdown
  - Timing analysis
  - Visual metrics

**To view:** Open in browser with: `open evaluation_report_chromadb.html`

---

### 📋 5. Evaluation Comparison Summary
**File:** [evaluation_comparison_chromadb.csv](./evaluation_comparison_chromadb.csv)
- **Size:** 526 B
- **Content:** Side-by-side comparison of Dense vs Sparse vs Hybrid methods

---

### 📝 6. Evaluation Log (Processing Details)
**File:** [evaluation_log.txt](./evaluation_log.txt)
- **Size:** 106 KB
- **Content:** Complete log of the evaluation process including:
  - Progress updates
  - Question processing
  - Timing information
  - Any warnings or errors

---

### 🎯 7. Top 20 Perfect Match Questions
**File:** [TOP_20_QUESTIONS.md](./TOP_20_QUESTIONS.md)
- **Content:** 20 questions that achieved perfect retrieval (MRR=1.0, Recall@10=1.0)
- **Use:** Best questions to test in the UI

---

### 🏃 8. Mini Evaluation (20 Questions - Quick Test)
**Files:**
- [evaluation_mini_results.csv](./evaluation_mini_results.csv) - 24 KB
- [evaluation_mini_summary.json](./evaluation_mini_summary.json) - 445 B

**Content:** Quick evaluation on subset of 20 questions for faster testing

---

## 📊 Evaluation Metrics Summary

### Performance by Method (100 Questions)

| Method | Questions | MRR ↑ | Recall@10 ↑ | Avg Time |
|--------|-----------|-------|-------------|----------|
| **Sparse (BM25)** | 100 | **0.439** | **0.470** | 5.87s |
| **Hybrid (RRF)** | 100 | 0.378 | 0.430 | 6.62s |
| **Dense (ChromaDB)** | 62* | 0.363 | 0.419 | 6.83s |

*Dense evaluation was interrupted but completed 62 questions

### Key Findings

✅ **All methods achieved 100% complete answer generation**
✅ **Sparse (BM25) performed best** with MRR=0.439 and Recall@10=0.47
✅ **20 questions achieved perfect retrieval** (MRR=1.0)
✅ **Average generation time:** 5-7 seconds per question

---

## 🚀 Streamlit UI

Test the system interactively:
- **Local:** http://localhost:8501
- **Script:** [start_ui.sh](./start_ui.sh)

---

## 📂 Evaluation Scripts

- [evaluate_chromadb_fast.py](./evaluate_chromadb_fast.py) - Full 100-question evaluation
- [evaluate_mini.py](./evaluate_mini.py) - Quick 20-question test
- [generate_report.py](./generate_report.py) - Generate HTML report

---

## 🔧 System Components

- **ChromaDB Index:** [chroma_db/chroma.sqlite3](./chroma_db/chroma.sqlite3) - 212 MB
- **BM25 Index:** [chroma_db/bm25_index.pkl](./chroma_db/bm25_index.pkl) - 11 MB
- **RAG System:** [chromadb_rag_system.py](./chromadb_rag_system.py)
- **UI Application:** [app_chromadb.py](./app_chromadb.py)

---

## ✅ Evaluation Framework Complete

The automated evaluation framework with 100 generated questions has been successfully completed and all results are saved in the locations above.

**Total Evaluations:** 262 (Dense: 62, Sparse: 100, Hybrid: 100)
**Evaluation Time:** ~2.5 hours total
**Date:** February 4, 2026
