# Hybrid RAG System - Retrieval-Augmented Generation

A comprehensive implementation of a Hybrid RAG system combining **Dense Vector Retrieval (FAISS)**, **Sparse Keyword Retrieval (BM25)**, and **Reciprocal Rank Fusion (RRF)** to answer questions from 500 Wikipedia articles.

## 🚀 Quick Start

**One-command setup (macOS/Linux):**
```bash
./run_all.sh
```

**One-command setup (Windows):**
```cmd
run_all.bat
```

This will automatically:
1. Set up virtual environment
2. Install all dependencies
3. Collect 500 Wikipedia articles
4. Build FAISS and BM25 indexes
5. Generate 100 evaluation questions
6. Run comprehensive evaluation
7. Optionally launch the UI

## 🎯 Project Overview

This project implements a state-of-the-art Hybrid RAG system that:
- Combines dense and sparse retrieval for superior performance
- Uses Reciprocal Rank Fusion to intelligently merge results
- Generates answers using Flan-T5 language model
- Includes comprehensive evaluation with 100 generated questions
- Features innovative evaluation techniques (ablation studies, error analysis, LLM-as-judge)

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User Query                              │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼───────┐
│ Dense Retrieval│   │Sparse Retrieval│
│  (FAISS + SE)  │   │     (BM25)     │
└───────┬────────┘   └────────┬───────┘
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Reciprocal Rank     │
        │ Fusion (RRF)        │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Top-N Chunks       │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  LLM Generation     │
        │   (Flan-T5)         │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Generated Answer  │
        └─────────────────────┘
```

## 🗂️ Project Structure

```
ConvAI_assingment_2/
├── data/
│   ├── fixed_urls.json          # 200 fixed Wikipedia URLs
│   ├── corpus.json              # Processed corpus with chunks
│   └── questions_100.json       # 100 evaluation questions
├── src/
│   ├── data_collection.py       # Wikipedia data collection
│   ├── rag_system.py            # Main RAG implementation
│   └── question_generation.py   # Question generation
├── evaluation/
│   ├── metrics.py               # Evaluation metrics (MRR, NDCG, BERTScore)
│   ├── innovative_eval.py       # Advanced evaluation features
│   └── pipeline.py              # Automated evaluation pipeline
├── models/
│   ├── faiss_index             # Dense vector index
│   └── bm25_index.pkl          # Sparse BM25 index
├── reports/
│   ├── evaluation_results.json  # Detailed results
│   ├── evaluation_results.csv   # Tabular results
│   ├── visualizations/          # Charts and plots
│   ├── ablation/                # Ablation study results
│   └── errors/                  # Error analysis
├── app.py                       # Streamlit UI
├── config.yaml                  # Configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional (recommended for faster processing)

### Setup Instructions

#### Option 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
./run_all.sh
```

**Windows:**
```cmd
run_all.bat
```

This single command will:
- Create virtual environment
- Install all dependencies
- Collect 500 Wikipedia articles
- Build FAISS and BM25 indexes
- Generate 100 evaluation questions
- Run complete evaluation
- Optionally launch the Streamlit UI

**Total time**: ~90-150 minutes (mostly automated)

#### Option 2: Manual Setup

1. **Clone/Download the repository**
```bash
cd ConvAI_assingment_2
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Option 3: Interactive Setup

For guided step-by-step setup:
```bash
python setup.py
```

## 📚 Usage Guide

### Quick Test

Run a quick test without full evaluation:
```bash
python quick_test.py
```

This will run 5 sample queries and show results.

### Step 1: Data Collection

Collect 200 fixed + 300 random Wikipedia URLs and process them into chunks:

```bash
# Generate fixed URLs first
python generate_fixed_urls.py

# Then collect all data
python src/data_collection.py
```

This creates:
- `data/fixed_urls.json` - 200 fixed URLs (remains constant)
- `data/corpus.json` - Processed corpus with all chunks

**Time**: ~30-60 minutes depending on internet speed

### Step 2: Build Indexes

Build dense (FAISS) and sparse (BM25) indexes:

```bash
python -c "
from src.rag_system import HybridRAGSystem
rag = HybridRAGSystem()
rag.load_corpus()
rag.build_dense_index()
rag.build_sparse_index()
"
```

This creates:
- `models/faiss_index` - Dense vector index
- `models/bm25_index.pkl` - Sparse keyword index

**Time**: ~10-20 minutes

### Step 3: Generate Evaluation Questions

Generate 100 diverse questions for evaluation:

```bash
python src/question_generation.py
```

This creates:
- `data/questions_100.json` - 100 Q&A pairs with metadata

**Time**: ~5-10 minutes

### Step 4: Run Evaluation Pipeline

Run the complete automated evaluation:

```bash
python evaluation/pipeline.py
```

This performs:
1. Evaluates all 100 questions
2. Calculates MRR, NDCG@5, BERTScore
3. Runs ablation study (dense vs sparse vs hybrid)
4. Performs error analysis
5. Generates visualizations
6. Saves comprehensive reports

**Output**: All results in `reports/` directory

**Time**: ~30-60 minutes

### Step 5: Launch Streamlit UI

Start the interactive web interface:

```bash
streamlit run app.py
```

Access at: http://localhost:8501

## 📊 Evaluation Metrics

### 1. Mean Reciprocal Rank (MRR) - **MANDATORY**

**Purpose**: Measures how quickly the system identifies the correct source document.

**Calculation**:
```
For each query:
  RR = 1/rank (if found), 0 (if not found)
MRR = Average of all RR scores
```

**Interpretation**:
- 1.0: Perfect - correct URL always ranked first
- 0.7-1.0: Excellent
- 0.5-0.7: Good
- < 0.5: Needs improvement

### 2. BERTScore F1 - **CUSTOM METRIC 1**

**Why Chosen**: Evaluates semantic similarity using contextual embeddings, capturing meaning beyond lexical matching.

**Calculation**:
1. Compute BERT embeddings for tokens
2. Calculate cosine similarity matrix
3. Greedy matching for optimal alignment
4. F1 = 2 * (Precision * Recall) / (Precision + Recall)

**Interpretation**:
- > 0.9: Excellent semantic match
- 0.8-0.9: Good match
- 0.7-0.8: Moderate match
- < 0.7: Poor match

### 3. NDCG@5 - **CUSTOM METRIC 2**

**Why Chosen**: Evaluates ranking quality considering both relevance and position. Critical for RAG as position affects context quality.

**Calculation**:
```
DCG@5 = Σ(i=1 to 5) [rel_i / log2(i+1)]
NDCG@5 = DCG@5 / IDCG@5
```

**Interpretation**:
- 1.0: Perfect ranking
- 0.8-1.0: Excellent
- 0.6-0.8: Good
- < 0.6: Needs improvement

## 🎨 Innovative Evaluation Features

1. **Ablation Study**: Compares dense-only, sparse-only, and hybrid performance
2. **Error Analysis**: Categorizes failures by type and question category
3. **LLM-as-Judge**: Uses LLM to evaluate answer quality
4. **Adversarial Testing**: Tests with negated and paraphrased questions
5. **Confidence Calibration**: Analyzes correlation between confidence and correctness
6. **Interactive Dashboard**: Real-time visualizations of all metrics

## 📈 Expected Results

Based on typical performance:

| Metric | Dense Only | Sparse Only | Hybrid (RRF) |
|--------|-----------|-------------|--------------|
| MRR | 0.45-0.60 | 0.40-0.55 | **0.55-0.70** |
| NDCG@5 | 0.50-0.65 | 0.45-0.60 | **0.60-0.75** |
| BERTScore F1 | 0.65-0.75 | 0.60-0.70 | **0.70-0.80** |

Hybrid approach typically outperforms individual methods by 10-20%.

## 🎯 Key Features

### Data Collection
- ✅ 200 fixed Wikipedia URLs (diverse topics)
- ✅ 300 random URLs per run
- ✅ Intelligent chunking (200-400 tokens, 50-token overlap)
- ✅ Metadata tracking (URL, title, chunk IDs)

### Retrieval System
- ✅ Dense retrieval with sentence-transformers
- ✅ Sparse retrieval with BM25
- ✅ Reciprocal Rank Fusion (k=60)
- ✅ Configurable top-K and top-N

### Generation
- ✅ Flan-T5-base for answer generation
- ✅ Context-aware prompting
- ✅ Configurable generation parameters

### Evaluation
- ✅ 100 diverse questions (factual, comparative, inferential, multi-hop)
- ✅ 3 comprehensive metrics (MRR, BERTScore, NDCG)
- ✅ Ablation studies
- ✅ Error analysis with categorization
- ✅ Rich visualizations

### User Interface
- ✅ Interactive Streamlit app
- ✅ Real-time query processing
- ✅ Source visualization with scores
- ✅ Performance metrics display
- ✅ Response time tracking

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
models:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  generation_model: "google/flan-t5-base"

retrieval:
  dense:
    top_k: 20
  sparse:
    top_k: 20
  rrf:
    k: 60
    final_top_n: 5
```

## 📝 Fixed URLs

The 200 fixed Wikipedia URLs cover diverse topics:

- **Science**: Physics, Chemistry, Biology, Astronomy, Geology
- **Technology**: AI, Computer Science, Robotics, Internet, Quantum Computing
- **History**: Ancient Egypt, Roman Empire, WWII, Renaissance
- **Geography**: Mountains, Rivers, Oceans, Countries
- **Arts**: Famous artists, Classical music, Literature
- **Sports**: Olympic Games, FIFA World Cup, Cricket
- **Philosophy**: Major philosophers, Ethics, Metaphysics
- **Mathematics**: Calculus, Linear Algebra, Statistics
- **Medicine**: Anatomy, Genetics, Immunology

Full list in `data/fixed_urls.json`

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size in `config.yaml` or use smaller model

### Issue: Slow Indexing
**Solution**: Use GPU if available, or reduce corpus size for testing

### Issue: Low Scores
**Solution**: 
- Check if questions match corpus topics
- Adjust RRF k parameter
- Try different embedding models

### Issue: Import Errors
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

## 📊 Sample Output

```
OVERALL RESULTS:
  MRR:           0.6234
  NDCG@5:        0.6891
  BERTScore F1:  0.7456
  Precision@5:   0.4200
  Recall@5:      0.5834
  ROUGE-L:       0.3987

Performance by Question Type:
  factual: MRR=0.72, NDCG=0.75
  comparative: MRR=0.58, NDCG=0.65
  inferential: MRR=0.51, NDCG=0.61
  multi_hop: MRR=0.49, NDCG=0.58
```

## 🎓 Academic Context

This project is designed for educational purposes as part of a Conversational AI assignment. It demonstrates:
- Modern RAG architecture
- Hybrid retrieval techniques
- Comprehensive evaluation methodologies
- Best practices in ML system development

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different models
- Add new evaluation metrics
- Improve the UI
- Optimize performance

## 📄 License

This project is for educational purposes only.

## 🙏 Acknowledgments

- **Sentence Transformers**: For embedding models
- **FAISS**: For efficient vector search
- **Rank-BM25**: For keyword retrieval
- **Hugging Face**: For LLM models
- **Streamlit**: For the UI framework

---

## 🚀 Quick Start Commands

```bash
# Full pipeline (run in order)
python src/data_collection.py
python -c "from src.rag_system import HybridRAGSystem; rag = HybridRAGSystem(); rag.load_corpus(); rag.build_dense_index(); rag.build_sparse_index()"
python src/question_generation.py
python evaluation/pipeline.py
streamlit run app.py
```

**Total setup time**: ~1-2 hours
**Total execution time**: ~2-3 hours

---

**Built with ❤️ for Conversational AI**
