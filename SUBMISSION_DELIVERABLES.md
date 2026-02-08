# SUBMISSION DELIVERABLES MAPPING

## Hybrid RAG System with Automated Evaluation

**GitHub Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)

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

## 1. EVALUATION COMPONENTS

### Question Generation

| Component | File | GitHub Link |
|-----------|------|-------------|
| Question Generation Script | src/question_generation.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/src/question_generation.py) |
| Dataset Creation | evaluation/create_dataset.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/create_dataset.py) |

### 100-Question Dataset (JSON)

| File | Description | GitHub Link |
|------|-------------|-------------|
| data/questions_100.json | 100 Q&A pairs for evaluation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/questions_100.json) |
| data/adversarial_questions.json | 30 adversarial questions | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/adversarial_questions.json) |

### Evaluation Pipeline

| Component | File | GitHub Link |
|-----------|------|-------------|
| Main Evaluation Script | evaluate_chromadb_fast.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluate_chromadb_fast.py) |
| Evaluation Pipeline | evaluation/pipeline.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/pipeline.py) |
| Run Evaluation | evaluation/run_evaluation.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/run_evaluation.py) |

### Metrics Implementation

| Metric | File | GitHub Link |
|--------|------|-------------|
| Core Metrics (MRR, Recall, F1) | evaluation/metrics.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/metrics.py) |
| Comprehensive Metrics | evaluation/comprehensive_metrics.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/comprehensive_metrics.py) |
| Novel Metrics | evaluation/novel_metrics.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/novel_metrics.py) |
| Innovative Evaluation | evaluation/innovative_eval.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/innovative_eval.py) |

### Innovative Components

| Component | Description | File | GitHub Link |
|-----------|-------------|------|-------------|
| Adversarial Testing | 30 adversarial questions | data/adversarial_questions.json | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/adversarial_questions.json) |
| Ablation Studies | Dense vs Sparse vs Hybrid | evaluation/run_evaluation.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/run_evaluation.py) |
| Error Analysis | Failure categorization | error_analysis.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/error_analysis.py) |
| Novel Metrics | Entity Coverage, Diversity, Hallucination | evaluation/novel_metrics.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation/novel_metrics.py) |

---

## 2. REPORT (PDF)

### Main Report

| File | Description | GitHub Link |
|------|-------------|-------------|
| Hybrid_RAG_Evaluation_Report.pdf | Complete PDF report | [Download](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/submission/05_reports/Hybrid_RAG_Evaluation_Report.pdf) |
| Hybrid_RAG_Evaluation_Report.md | Markdown version | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/submission/05_reports/Hybrid_RAG_Evaluation_Report.md) |

### Architecture Diagram

| File | Description | GitHub Link |
|------|-------------|-------------|
| docs/architecture_diagram.png | System architecture | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/architecture_diagram.png) |
| docs/data_flow_diagram.png | Data flow diagram | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/data_flow_diagram.png) |

### Evaluation Results with Tables/Visualizations

| File | Description | GitHub Link |
|------|-------------|-------------|
| evaluation_results_chromadb.csv | Detailed results (300 rows) | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_results_chromadb.csv) |
| evaluation_summary_chromadb.json | Summary statistics | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_summary_chromadb.json) |
| docs/retrieval_heatmap.png | Performance heatmap | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/retrieval_heatmap.png) |

### Innovative Approach Description

| File | Description | GitHub Link |
|------|-------------|-------------|
| docs/NEW_FEATURES.md | New features documentation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/NEW_FEATURES.md) |
| docs/METRIC_JUSTIFICATION.md | Metric justification | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/METRIC_JUSTIFICATION.md) |

### Ablation Studies

| Component | Location | GitHub Link |
|-----------|----------|-------------|
| Dense vs Sparse vs Hybrid comparison | evaluation_results_chromadb.csv | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_results_chromadb.csv) |
| Method comparison in report | Hybrid_RAG_Evaluation_Report.pdf | [Download](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/submission/05_reports/Hybrid_RAG_Evaluation_Report.pdf) |

### Error Analysis

| File | Description | GitHub Link |
|------|-------------|-------------|
| error_analysis.py | Error analysis script | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/error_analysis.py) |
| docs/error_analysis_charts.png | Error analysis visualizations | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/docs/error_analysis_charts.png) |
| submission/06_documentation/ERROR_ANALYSIS.md | Error analysis documentation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/submission/06_documentation/ERROR_ANALYSIS.md) |

### System Screenshots (3+)

| Screenshot | Description | GitHub Link |
|------------|-------------|-------------|
| 1_Hybrid_full_page.png | Hybrid retrieval mode | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/1_Hybrid_full_page.png) |
| 2_Dense_full_page.png | Dense retrieval mode | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/2_Dense_full_page.png) |
| 3_Sparse_full_page.png | Sparse retrieval mode | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/3_Sparse_full_page.png) |
| Data_load_retrieval_method.png | Data loading interface | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/Data_load_retrieval_method.png) |
| project_SoureCode.png | Project source code | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/screenshots/project_SoureCode.png) |

---

## 3. INTERFACE

### Streamlit Application

| Component | File | GitHub Link |
|-----------|------|-------------|
| Streamlit UI | app_chromadb.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/app_chromadb.py) |
| FastAPI Backend | api_chromadb.py | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/api_chromadb.py) |

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation.git
cd Hybrid_RAG_System_with_Automated_Evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build vector database (if not exists)
python build_chromadb_system.py

# Run Streamlit Dashboard
streamlit run app_chromadb.py

# Run FastAPI Server (alternative)
uvicorn api_chromadb:app --reload
```

---

## 4. README.md

### Main Documentation

| File | Description | GitHub Link |
|------|-------------|-------------|
| README.md | Main project documentation | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/README.md) |

### Contents of README.md

| Section | Description |
|---------|-------------|
| Installation Steps | Complete setup guide |
| Dependencies | requirements.txt reference |
| Run Instructions | System and evaluation commands |
| Fixed URLs Reference | Link to fixed_urls.json |

### Dependencies

| File | Description | GitHub Link |
|------|-------------|-------------|
| requirements.txt | Python dependencies | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/requirements.txt) |
| config.yaml | System configuration | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/config.yaml) |

---

## 5. DATA FILES

### Fixed URLs (200 Wikipedia URLs)

| File | Description | GitHub Link |
|------|-------------|-------------|
| data/fixed_urls.json | 200 fixed Wikipedia URLs in JSON format | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/fixed_urls.json) |

### Preprocessed Corpus

| File | Description | GitHub Link |
|------|-------------|-------------|
| data/corpus.json | Preprocessed corpus (7,519 chunks) | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/corpus.json) |

### Vector Database

| Location | Description | Regeneration Instructions |
|----------|-------------|---------------------------|
| chroma_db/ | ChromaDB vector database | Run `python build_chromadb_system.py` |

**Regeneration Command:**
```bash
python build_chromadb_system.py
```

### 100-Question Dataset

| File | Description | GitHub Link |
|------|-------------|-------------|
| data/questions_100.json | 100 Q&A evaluation pairs | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/data/questions_100.json) |

### Evaluation Results

| File | Description | GitHub Link |
|------|-------------|-------------|
| evaluation_results_chromadb.csv | Detailed results (300 rows) | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_results_chromadb.csv) |
| evaluation_summary_chromadb.json | Summary statistics | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/blob/main/evaluation_summary_chromadb.json) |

**Regeneration Command:**
```bash
python evaluate_chromadb_fast.py
```

---

## 6. SUBMISSION FOLDER STRUCTURE

| Folder | Contents | GitHub Link |
|--------|----------|-------------|
| submission/01_source_code/ | All Python source files | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/01_source_code) |
| submission/02_data/ | Corpus and question datasets | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/02_data) |
| submission/03_vector_database/ | ChromaDB files | Refer instructions to setup vectorDB |
| submission/04_evaluation_results/ | CSV and JSON results | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/04_evaluation_results) |
| submission/05_reports/ | PDF and MD reports | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/05_reports) |
| submission/06_documentation/ | Documentation files | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/06_documentation) |
| submission/07_visualizations/ | Charts and diagrams | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/07_visualizations) |
| submission/08_screenshots/ | UI screenshots | [View](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation/tree/main/submission/08_screenshots) |

---

## Quick Reference

| Requirement | Location |
|-------------|----------|
| Question Generation Script | src/question_generation.py |
| 100-Question Dataset (JSON) | data/questions_100.json |
| Evaluation Pipeline | evaluate_chromadb_fast.py |
| Metrics Implementation | evaluation/metrics.py |
| Innovative Components | evaluation/novel_metrics.py |
| PDF Report | submission/05_reports/Hybrid_RAG_Evaluation_Report.pdf |
| Architecture Diagram | docs/architecture_diagram.png |
| Evaluation Results | evaluation_results_chromadb.csv |
| Ablation Studies | evaluation_results_chromadb.csv (3 methods) |
| Error Analysis | error_analysis.py, docs/error_analysis_charts.png |
| System Screenshots (3+) | screenshots/ (5 screenshots) |
| Streamlit Interface | app_chromadb.py |
| README.md | README.md |
| Fixed 200 URLs (JSON) | data/fixed_urls.json |
| Preprocessed Corpus | data/corpus.json |
| Vector Database | chroma_db/ (regenerate with build script) |

---

**Last Updated:** February 8, 2026  
**Repository:** [https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation](https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation)
