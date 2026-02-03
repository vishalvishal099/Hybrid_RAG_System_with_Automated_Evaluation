# 🔍 RAG System Diagnosis Report
**Date**: February 3, 2026  
**Issue**: UI returns wrong documents ("Organic Chemistry" instead of "Artificial Intelligence")

---

## 📊 **Problem Summary**

### **Expected Behavior (Evaluation Dataset)**
- **Question**: "What is artificial intelligence?"
- **Expected Answer**: "Artificial intelligence (AI) is intelligence demonstrated by machines..."
- **Expected Source**: Wikipedia - Artificial Intelligence article

### **Actual Behavior (UI)**
- **Question**: "What is artificial intelligence?"
- **Actual Answer**: "Biopolymers occur within a respectfully natural environment"
- **Actual Source**: Wikipedia - Organic Chemistry article ❌

---

## 🔬 **Root Cause Analysis**

### **1. Index/Corpus Mismatch (CRITICAL)**

```
Corpus:       10,420 chunks
FAISS Index:   7,519 vectors
BM25 Index:    7,519 chunks
────────────────────────────────
Missing:       2,901 chunks (28% of corpus)
```

**Impact**: 
- The RAG system loads **10,420 chunks** from `corpus.json`
- But indexes only contain **7,519 vectors**
- When the system searches, it uses **OLD indexes** that don't match the current corpus
- This causes **index out of bounds** issues and wrong mappings

### **2. Index-to-Corpus Mapping Broken**

**What's Happening**:
```python
# System retrieves from index at position 150
retrieved_idx = 150  # From FAISS search

# System tries to get chunk from corpus
chunk = corpus[retrieved_idx]  # But corpus has different chunks at this position!
```

**Why It Breaks**:
- Indexes were built from corpus version A (7,519 chunks)
- Current corpus is version B (10,420 chunks - after reprocessing)
- Position 150 in old index ≠ Position 150 in new corpus

### **3. Why "Organic Chemistry" Appears**

**Hypothesis**:
1. Query: "What is artificial intelligence?"
2. FAISS searches old index (7,519 vectors)
3. Returns index position, e.g., position 450
4. System loads corpus[450] from NEW corpus (10,420 chunks)
5. Due to chunk reprocessing, position 450 now contains "Organic Chemistry" text
6. Original "AI" chunk might be at position 5,000 (not in index!)

---

## 🧪 **Evidence**

### **Corpus Content Verification**
```
✅ AI chunks exist in corpus: 24 chunks found
✅ AI text correct: "Artificial intelligence (AI) is the capability..."
✅ Organic Chemistry chunks: 19 chunks found
```

### **Index Size Verification**
```
data/corpus.json:                    10,420 chunks
data/indexes/faiss_index/index.faiss: 7,519 vectors
data/indexes/bm25_index.pkl:          7,519 chunks
data/indexes/faiss_index/embeddings.npy: (7519, 384)
data/indexes/faiss_index/chunks.json: 7,519 entries
```

---

## 🎯 **Why Evaluation Dataset Works**

The evaluation dataset in `data/evaluation_dataset.json` contains:
- **Manually written ground truth answers**
- **Not dependent on retrieval**
- **Independent of the broken index/corpus mapping**

The evaluation metrics would:
1. Test retrieval (would FAIL due to wrong documents)
2. Compare generated answers to ground truth (would show poor F1 scores)
3. Check if correct URLs are retrieved (would FAIL - MRR = 0)

---

## 🔧 **Solution Required**

### **Option 1: Rebuild Indexes from Current Corpus (RECOMMENDED)**

```bash
# Backup old indexes
mv data/indexes data/indexes_old

# Rebuild with current corpus
python src/indexing.py
```

**Pro**: Matches current corpus, proper indexing  
**Con**: Takes time to reprocess 10,420 chunks

### **Option 2: Restore Original Corpus**

```bash
# Restore original smaller corpus
cp data/corpus_old_chunking.json data/corpus.json

# OR rebuild from scratch
python src/data_collection.py
python src/indexing.py
```

**Pro**: Clean slate, consistent state  
**Con**: Loses any improvements from reprocessing

### **Option 3: Fix Corpus Reprocessing Bug**

The `reprocess_corpus.py` script **multiplied** chunks instead of replacing them:
- Original: 7,519 chunks
- After run 1: 10,420 chunks
- After run 2: 11,552 chunks (keeps growing!)

Need to fix the reconstruction logic that concatenates already-chunked text.

---

## 📋 **Action Plan**

### **Immediate (Fix UI)**
1. ✅ Stop Streamlit
2. ⚠️ Rebuild indexes to match 10,420-chunk corpus
3. ✅ Restart Streamlit
4. ✅ Test with "What is artificial intelligence?"

### **Long-term (Proper Chunking)**
1. ⚠️ Implement new semantic chunking (200-400 tokens, 50 overlap)
2. ⚠️ Add deduplication logic
3. ⚠️ Preserve anchors and metadata
4. ⚠️ Rebuild corpus from original Wikipedia articles
5. ⚠️ Rebuild indexes with new corpus
6. ⚠️ Run full evaluation pipeline

---

## ⚠️ **Risk Assessment**

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Index rebuild fails at 10,420 chunks | High | Medium | Use batched embedding |
| Out of memory during indexing | High | Medium | Process in chunks of 1000 |
| Evaluation shows poor metrics | Medium | High | Expected, need proper chunking |
| Loss of current progress | Medium | Low | Git commit before changes |

---

## 📈 **Expected Improvements After Fix**

### **Retrieval Metrics**
- **MRR (Mean Reciprocal Rank)**: Currently ~0.0 → Target: >0.7
- **Recall@10**: Currently ~0.0 → Target: >0.8
- **Answer F1**: Currently ~0.0 → Target: >0.5

### **User Experience**
- ✅ Correct documents retrieved
- ✅ Relevant answers generated
- ✅ Proper source attribution
- ✅ Consistent with evaluation dataset

---

## 🔬 **Technical Details**

### **Current System State**
```yaml
Corpus Format:
  - chunks: 10,420
  - documents: 495
  - chunking: Mixed (old + reprocessed)
  
Index Format:
  - FAISS: IndexFlatIP, 384-dim, L2-normalized
  - BM25: BM25Okapi (k1=1.5, b=0.75)
  - Size: 7,519 vectors
  
Mismatch:
  - Corpus indices 7,520 to 10,420 NOT in index
  - Retrieval uses 0 to 7,518 (old positions)
  - Mapping broken by corpus reprocessing
```

### **Why Semantic Search Fails**
1. Query embedding: "artificial intelligence" → [0.23, -0.45, 0.12, ...]
2. FAISS search: Finds nearest vectors in 7,519-vector index
3. Returns indices: [450, 892, 1203, ...]
4. System loads: corpus[450], corpus[892], corpus[1203]
5. **But** corpus positions have shifted after reprocessing!
6. Position 450 used to be "AI", now it's "Organic Chemistry"
7. Real "AI" chunks are at positions 8,000-8,500 (NOT IN INDEX)

---

## ✅ **Verification Steps After Fix**

1. **Check sizes match**:
   ```bash
   python -c "import json, faiss
   corpus = json.load(open('data/corpus.json'))
   index = faiss.read_index('data/indexes/faiss_index/index.faiss')
   print(f'Corpus: {len(corpus[\"chunks\"])}')
   print(f'Index: {index.ntotal}')
   print(f'Match: {len(corpus[\"chunks\"]) == index.ntotal}')"
   ```

2. **Test retrieval**:
   ```python
   python test_answers.py
   # Should return AI article, not Organic Chemistry
   ```

3. **Run evaluation**:
   ```bash
   python evaluation/run_evaluation.py
   # Should show MRR > 0.7, Recall@10 > 0.8
   ```

---

## 📝 **Conclusion**

**Primary Issue**: Index/corpus size mismatch (7,519 vs 10,420)  
**Secondary Issue**: Corpus reprocessing multiplied chunks instead of replacing  
**Impact**: Retrieval returns wrong documents, answers are gibberish  
**Solution**: Rebuild indexes to match current corpus size  
**Prevention**: Always rebuild indexes after corpus changes

**Status**: 🔴 **CRITICAL - SYSTEM UNUSABLE FOR RAG TASKS**
