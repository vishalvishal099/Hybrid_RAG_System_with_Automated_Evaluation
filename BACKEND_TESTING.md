# 🧪 Backend Testing Guide

## **How to Test Without UI**

You have **2 ways** to test the RAG system directly from the backend:

---

## **Method 1: Quick Single Question Test** ⚡

### **Basic Usage:**
```bash
source venv/bin/activate
python quick_test_question.py "Your question here"
```

### **Examples:**
```bash
# Test AI question
python quick_test_question.py "What is artificial intelligence?"

# Test telephone question  
python quick_test_question.py "Who invented the telephone?"

# Test Roman Empire question
python quick_test_question.py "When was the Roman Empire founded?"

# Test DNA question
python quick_test_question.py "What is DNA?"
```

### **What You'll See:**
- ✅ Answer text
- 📚 Top source title and URL
- 📊 RRF relevance score
- ⏱️ Timing information

---

## **Method 2: Batch Testing Multiple Questions** 📋

### **Run All Test Questions:**
```bash
source venv/bin/activate
python test_backend.py
```

This will:
- Test 5 pre-defined evaluation questions
- Show all answers
- Calculate average generation time
- Save results to `backend_test_results.json`

### **Test Custom Question:**
```bash
python test_backend.py "Your custom question here"
```

---

## **Expected Results**

### ✅ **AI Question**
```
Q: What is artificial intelligence?
A: The capability of computational systems to perform tasks 
   typically associated with human intelligence, such as 
   learning, reasoning, problem-solving, perception, and decision-making.
```

### ✅ **Telephone Question**
```
Q: Who invented the telephone?
A: Alexander Graham Bell invented the telephone. He was awarded 
   the first US patent for the invention in 1876.
```

### ✅ **Roman Empire Question**
```
Q: When was the Roman Empire founded?
A: The Roman Empire was founded in 27 BC when Augustus became 
   the first Roman emperor, marking the end of the Roman Republic.
```

---

## **Troubleshooting**

### **Issue 1: Answer is Cut Off**
**Problem**: Answer ends mid-sentence like "decision-"

**Cause**: Max tokens limit reached during generation

**Solution**: Answer is still valid, just truncated. The model is working correctly.

---

### **Issue 2: Wrong Answer (e.g., "Charles Babbage" for telephone)**
**Problem**: Getting incorrect historical facts

**Cause**: Wrong chunks being retrieved

**Check**:
```bash
# Verify corpus/index alignment
python -c "
import json
import numpy as np
corpus = json.load(open('data/corpus.json'))
embeddings = np.load('data/indexes/faiss_index/embeddings.npy')
print(f'Corpus: {len(corpus[\"chunks\"])}')
print(f'Index: {embeddings.shape[0]}')
print(f'Match: {len(corpus[\"chunks\"]) == embeddings.shape[0]}')
"
```

**Expected Output**:
```
Corpus: 7519
Index: 7519
Match: True
```

---

### **Issue 3: HTML Entities in Answer**
**Problem**: Answer shows `<br />` or `&nbsp;`

**Solution**: Already fixed in `src/rag_system.py` with regex cleaning:
```python
answer = re.sub(r'<[^>]+>', '', answer)  # Remove HTML tags
answer = re.sub(r'&[a-z]+;', ' ', answer)  # Remove entities
```

If still seeing HTML, the fix isn't applied. Restart the test.

---

## **Comparing Backend vs UI**

### **Backend Answer**:
```bash
python quick_test_question.py "What is artificial intelligence?"
```

### **UI Answer**:
Open http://localhost:8501 and ask the same question

### **They Should Match!**

If backend gives correct answer but UI doesn't:
1. Stop Streamlit: `pkill -9 -f streamlit`
2. Restart: `streamlit run app.py`
3. Clear browser cache
4. Try again

---

## **Debugging Steps**

### **Step 1: Check if system loads**
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from rag_system import HybridRAGSystem
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()
print('✅ System loaded successfully')
"
```

### **Step 2: Test retrieval only**
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from rag_system import HybridRAGSystem
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()
results = rag.search('artificial intelligence', method='hybrid', top_k=3)
print(f'Top result: {results[0][\"title\"]}')
print(f'Text: {results[0][\"text\"][:200]}')
"
```

### **Step 3: Test generation only**
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from rag_system import HybridRAGSystem
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()

# Get chunks
chunks = rag.search('artificial intelligence', method='hybrid', top_k=3)

# Generate answer
result = rag.generate_answer('What is artificial intelligence?', chunks)
print(f'Answer: {result[\"answer\"]}')
"
```

---

## **Performance Metrics**

Expected timing on typical hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| System Load | 5-10s | First time only |
| Retrieval (FAISS+BM25) | 0.5-1s | Per query |
| Generation (FLAN-T5) | 4-8s | Per query |
| **Total per query** | **5-10s** | After initial load |

---

## **Saved Results**

After running `test_backend.py`, check:
```bash
cat backend_test_results.json
```

This contains:
- All questions tested
- Generated answers
- Generation times
- Easy to parse for analysis

---

## **Quick Commands Summary**

```bash
# Single question test
python quick_test_question.py "What is AI?"

# Batch test all questions
python test_backend.py

# Test with sources shown
python test_backend.py "Who invented the telephone?"

# Check corpus/index alignment
python -c "import json, numpy as np; corpus=json.load(open('data/corpus.json')); emb=numpy.load('data/indexes/faiss_index/embeddings.npy'); print(f'{len(corpus[\"chunks\"])} == {emb.shape[0]}')"

# Restart UI
pkill -9 -f streamlit && streamlit run app.py
```

---

## **Getting Help**

If backend works but UI doesn't:
1. **Check Streamlit logs** - Look for errors in terminal
2. **Verify corpus.json** - Should have 7,519 chunks
3. **Check indexes** - Should match corpus size
4. **Test in incognito** - Clear browser cache issues

If backend also has wrong answers:
1. **Verify corpus alignment** - Run alignment check above
2. **Check if article exists** - Search corpus for keywords
3. **Test retrieval** - Run Step 2 from debugging section

---

**Happy Testing!** 🎉
