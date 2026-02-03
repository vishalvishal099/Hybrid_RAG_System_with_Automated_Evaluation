# Quick Reference Guide

## 🚀 Quick Start (5 Commands)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate fixed URLs (optional - run if data/fixed_urls.json doesn't exist)
python generate_fixed_urls.py

# 3. Collect data
python src/data_collection.py

# 4. Build indexes & generate questions
python -c "from src.rag_system import HybridRAGSystem; rag = HybridRAGSystem(); rag.load_corpus(); rag.build_dense_index(); rag.build_sparse_index()"
python src/question_generation.py

# 5. Run evaluation & launch UI
python evaluation/pipeline.py
streamlit run app.py
```

## 📁 Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `config.yaml` | Configuration | Modify parameters |
| `src/data_collection.py` | Collect Wikipedia data | First time setup |
| `src/rag_system.py` | Main RAG system | Core functionality |
| `src/question_generation.py` | Generate questions | Create evaluation set |
| `evaluation/pipeline.py` | Run evaluation | Test system performance |
| `app.py` | Streamlit UI | Interactive testing |
| `setup.py` | Automated setup | Guided installation |

## 🎯 Common Tasks

### Test Single Query
```python
from src.rag_system import HybridRAGSystem

rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()

response = rag.query("What is AI?", method="hybrid")
print(response['answer'])
```

### Evaluate Custom Questions
```python
from evaluation.metrics import RAGEvaluator

evaluator = RAGEvaluator()
questions = [...]  # Your questions
rag = HybridRAGSystem()

for q in questions:
    response = rag.query(q['question'])
    result = evaluator.evaluate_single_query(q, response)
    print(f"MRR: {result['metrics']['mrr']:.3f}")
```

### Change Models
Edit `config.yaml`:
```yaml
models:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"  # Better quality
  generation_model: "google/flan-t5-large"  # Larger model
```

### Adjust Retrieval
Edit `config.yaml`:
```yaml
retrieval:
  dense:
    top_k: 30  # More dense results
  sparse:
    top_k: 30  # More sparse results
  rrf:
    k: 60  # RRF constant
    final_top_n: 10  # More context for LLM
```

## 📊 Understanding Metrics

| Metric | Range | Good Score | What It Measures |
|--------|-------|------------|------------------|
| MRR | 0-1 | >0.6 | How fast correct URL is found |
| NDCG@5 | 0-1 | >0.7 | Quality of top-5 ranking |
| BERTScore | 0-1 | >0.75 | Semantic similarity of answer |
| Precision@5 | 0-1 | >0.4 | Relevant docs in top-5 |
| Recall@5 | 0-1 | >0.5 | Coverage of relevant docs |

## 🐛 Troubleshooting

### Out of Memory
```yaml
# Reduce batch size in config.yaml
retrieval:
  dense:
    batch_size: 16  # Add this line
```

### Slow Processing
- Use GPU if available
- Reduce corpus size for testing
- Use smaller models

### Import Errors
```bash
pip install --upgrade -r requirements.txt
python -m nltk.downloader punkt stopwords
```

### Low Scores
1. Check question-corpus alignment
2. Try different RRF k values
3. Increase top_k in retrieval

## 📈 Performance Tips

### Speed Up
- Use GPU: Set `CUDA_VISIBLE_DEVICES=0`
- Smaller models: Use `distilgpt2` or `t5-small`
- Reduce corpus: Test with 100 URLs first

### Improve Quality
- Better embeddings: Use `all-mpnet-base-v2`
- Larger LLM: Use `flan-t5-large` or `flan-t5-xl`
- More context: Increase `final_top_n` to 10

## 🔧 Configuration Presets

### Fast Mode (Testing)
```yaml
models:
  embedding_model: "all-MiniLM-L6-v2"
  generation_model: "google/flan-t5-small"
retrieval:
  rrf:
    final_top_n: 3
```

### Quality Mode (Production)
```yaml
models:
  embedding_model: "all-mpnet-base-v2"
  generation_model: "google/flan-t5-base"
retrieval:
  rrf:
    final_top_n: 7
```

### Balanced Mode (Recommended)
```yaml
models:
  embedding_model: "all-MiniLM-L6-v2"
  generation_model: "google/flan-t5-base"
retrieval:
  rrf:
    final_top_n: 5
```

## 📝 File Locations

```
data/
├── fixed_urls.json         # 200 fixed URLs
├── corpus.json             # ~500MB (all processed data)
└── questions_100.json      # Evaluation questions

models/
├── faiss_index            # ~100-200MB (dense index)
└── bm25_index.pkl         # ~50-100MB (sparse index)

reports/
├── evaluation_results.json
├── evaluation_results.csv
├── visualizations/*.png
└── errors/error_analysis.json
```

## 🎨 Streamlit UI Tips

### Keyboard Shortcuts
- `Ctrl/Cmd + R`: Refresh
- `Ctrl/Cmd + K`: Clear cache

### URL Format
- Local: `http://localhost:8501`
- Network: `http://<your-ip>:8501`

### Deploy to Cloud
```bash
# Streamlit Cloud
streamlit run app.py --server.port 8501

# Heroku
# Add Procfile: web: streamlit run app.py --server.port $PORT
```

## 📦 Export Results

### JSON
```python
import json
with open('reports/evaluation_results.json', 'r') as f:
    results = json.load(f)
```

### CSV (Excel-ready)
```python
import pandas as pd
df = pd.read_csv('reports/evaluation_results.csv')
df.to_excel('results.xlsx', index=False)
```

### Visualizations
All charts saved as high-res PNG (300 DPI) in `reports/visualizations/`

## 🔍 Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check specific components:
```python
# Test data collection
from src.data_collection import WikipediaDataCollector
collector = WikipediaDataCollector()
collector.load_corpus()

# Test retrieval
from src.rag_system import HybridRAGSystem
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()
results = rag.retrieve("test query", method="hybrid")
print(results)

# Test evaluation
from evaluation.metrics import RAGEvaluator
evaluator = RAGEvaluator()
mrr = evaluator.calculate_mrr(["url1", "url2"], ["url1"])
print(f"MRR: {mrr}")
```

## 💡 Pro Tips

1. **Start Small**: Test with 50 URLs before full 500
2. **Cache Models**: Models downloaded once, reused
3. **Use Setup Script**: `python setup.py` for guided setup
4. **Check Reports**: Visualizations show what's working
5. **Iterate**: Adjust config based on evaluation results

## 📚 Additional Resources

- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **BM25**: https://en.wikipedia.org/wiki/Okapi_BM25
- **Streamlit**: https://docs.streamlit.io/

## 🆘 Getting Help

1. Check error messages carefully
2. Review `PROJECT_SUMMARY.md` for architecture
3. Read `README.md` for detailed docs
4. Examine `demo_notebook.md` for examples
5. Look at inline code comments

---

**Last Updated**: January 30, 2026  
**Version**: 1.0  
**Status**: Production Ready
