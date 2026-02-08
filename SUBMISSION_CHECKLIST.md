# Submission Checklist

## 📋 Pre-Submission Verification

### ✅ Required Files

- [ ] **Code Files**
  - [ ] `src/data_collection.py` (Wikipedia scraper)
  - [ ] `src/rag_system.py` (Hybrid RAG implementation)
  - [ ] `src/question_generation.py` (Q&A generation)
  - [ ] `evaluation/metrics.py` (MRR, BERTScore, NDCG)
  - [ ] `evaluation/innovative_eval.py` (Advanced features)
  - [ ] `evaluation/pipeline.py` (Automated evaluation)
  - [ ] `app.py` (Streamlit UI)

- [ ] **Configuration Files**
  - [ ] `config.yaml` (System configuration)
  - [ ] `requirements.txt` (Dependencies)
  - [ ] `.gitignore` (Git ignore rules)

- [ ] **Data Files**
  - [ ] `data/fixed_urls.json` (200 fixed Wikipedia URLs)
  - [ ] `data/corpus.json` (Processed corpus) OR instructions to regenerate
  - [ ] `data/questions_100.json` (100 evaluation questions)

- [ ] **Model Files**
  - [ ] `models/faiss_index` OR instructions to build
  - [ ] `models/bm25_index.pkl` OR instructions to build

- [ ] **Evaluation Results**
  - [ ] `reports/evaluation_results.json`
  - [ ] `reports/evaluation_results.csv`
  - [ ] `reports/metric_explanations.txt`
  - [ ] `reports/visualizations/*.png` (at least 5 charts)

- [ ] **Documentation**
  - [ ] `README.md` (Comprehensive documentation)
  - [ ] `PROJECT_SUMMARY.md` (Project completion summary)
  - [ ] `ARCHITECTURE.md` (System architecture)
  - [ ] `QUICK_REFERENCE.md` (Quick reference guide)

- [ ] **Screenshots**
  - [ ] UI showing query input
  - [ ] UI showing generated answer
  - [ ] UI showing retrieved sources with scores
  - [ ] Charts/visualizations from evaluation

### ✅ Code Quality

- [ ] All Python files have docstrings
- [ ] Functions are well-commented
- [ ] No syntax errors
- [ ] No critical lint warnings
- [ ] Consistent code style
- [ ] Meaningful variable names
- [ ] Error handling implemented

### ✅ Functionality

- [ ] **Data Collection Works**
  - [ ] Collects 200 fixed URLs
  - [ ] Collects 300 random URLs
  - [ ] Chunks text properly (200-400 tokens, 50-token overlap)
  - [ ] Saves to corpus.json

- [ ] **Dense Retrieval Works**
  - [ ] Loads embedding model
  - [ ] Builds FAISS index
  - [ ] Retrieves top-K chunks
  - [ ] Returns similarity scores

- [ ] **Sparse Retrieval Works**
  - [ ] Implements BM25
  - [ ] Tokenizes properly
  - [ ] Retrieves top-K chunks
  - [ ] Returns BM25 scores

- [ ] **RRF Works**
  - [ ] Combines dense and sparse results
  - [ ] Uses correct formula: 1/(k+rank)
  - [ ] Returns top-N by RRF score

- [ ] **Generation Works**
  - [ ] Loads LLM (Flan-T5)
  - [ ] Concatenates context
  - [ ] Generates coherent answers

- [ ] **Evaluation Works**
  - [ ] Calculates MRR correctly (URL-level)
  - [ ] Calculates BERTScore
  - [ ] Calculates NDCG@5
  - [ ] Generates comprehensive reports

- [ ] **UI Works**
  - [ ] Accepts query input
  - [ ] Displays generated answer
  - [ ] Shows retrieved sources
  - [ ] Displays all scores (dense, sparse, RRF)
  - [ ] Shows response time

### ✅ Evaluation Metrics

- [ ] **MRR (Mandatory)**
  - [ ] Implemented correctly
  - [ ] URL-level (not chunk-level)
  - [ ] Detailed justification written
  - [ ] Calculation method explained
  - [ ] Interpretation guidelines provided

- [ ] **BERTScore (Custom Metric 1)**
  - [ ] Implemented correctly
  - [ ] Justification: Why chosen for RAG
  - [ ] Calculation: BERT embeddings → similarity → F1
  - [ ] Interpretation: Score ranges explained

- [ ] **NDCG@5 (Custom Metric 2)**
  - [ ] Implemented correctly
  - [ ] Justification: Why ranking matters
  - [ ] Calculation: DCG/IDCG formula
  - [ ] Interpretation: What scores mean

### ✅ Innovation

- [ ] **Ablation Study**
  - [ ] Compares dense vs sparse vs hybrid
  - [ ] Shows performance differences
  - [ ] Includes visualizations

- [ ] **Error Analysis**
  - [ ] Categorizes failures
  - [ ] Analyzes by question type
  - [ ] Provides examples
  - [ ] Includes visualizations

- [ ] **Additional Features** (at least 2 more)
  - [ ] LLM-as-Judge
  - [ ] Adversarial Testing
  - [ ] Confidence Calibration
  - [ ] Performance visualizations

### ✅ Documentation

- [ ] **README.md Complete**
  - [ ] Installation instructions
  - [ ] Usage guide with examples
  - [ ] Architecture diagram
  - [ ] Metric explanations
  - [ ] Fixed URLs list
  - [ ] Troubleshooting section

- [ ] **Code Documentation**
  - [ ] All modules have docstrings
  - [ ] All classes documented
  - [ ] All functions documented
  - [ ] Complex logic explained

- [ ] **Metric Documentation**
  - [ ] Why each metric was chosen
  - [ ] How each metric is calculated
  - [ ] How to interpret results
  - [ ] Examples provided

### ✅ Testing

- [ ] **Manual Tests**
  - [ ] Run data collection successfully
  - [ ] Build indexes without errors
  - [ ] Generate 100 questions
  - [ ] Run evaluation pipeline
  - [ ] Launch UI and test queries

- [ ] **Verify Outputs**
  - [ ] evaluation_results.json has all fields
  - [ ] evaluation_results.csv opens in Excel
  - [ ] Visualizations are clear and labeled
  - [ ] UI displays correctly

### ✅ Performance

- [ ] System processes queries in reasonable time (<5s per query)
- [ ] Evaluation completes within 2 hours
- [ ] UI is responsive
- [ ] No memory leaks or crashes

### ✅ Final Package

- [ ] **Create ZIP File**
  - [ ] Name: `Group_<Number>_Hybrid_RAG.zip`
  - [ ] Includes all required files
  - [ ] Organized folder structure
  - [ ] < 200MB (exclude large model files if needed)

- [ ] **Test ZIP File**
  - [ ] Extract to new location
  - [ ] Run setup instructions
  - [ ] Verify everything works

## 🎯 Scoring Self-Assessment

### Part 1: Hybrid RAG
- Dense Retrieval (2): ___/2
- Sparse Retrieval (2): ___/2
- RRF (2): ___/2
- Generation (2): ___/2
- UI (2): ___/2
**Subtotal**: ___/10

### Part 2: Evaluation
- Questions (1): ___/1
- MRR (2): ___/2
- Custom Metrics (4): ___/4
- Innovation (4): ___/4
**Subtotal**: ___/10

**TOTAL**: ___/20

## 📝 Pre-Submission Checklist

1. [ ] All code runs without errors
2. [ ] All dependencies listed in requirements.txt
3. [ ] README has clear instructions
4. [ ] Fixed URLs file exists and has 200 URLs
5. [ ] 100 questions generated with diverse types
6. [ ] 3 metrics implemented (MRR + 2 custom)
7. [ ] Each metric has justification + calculation + interpretation
8. [ ] Ablation study completed
9. [ ] Error analysis completed
10. [ ] At least 5 visualizations generated
11. [ ] UI works and shows all required information
12. [ ] Screenshots included (at least 3)
13. [ ] All files properly named
14. [ ] Code is well-commented
15. [ ] Documentation is comprehensive
16. [ ] ZIP file created and tested
17. [ ] File size is reasonable
18. [ ] Submission follows naming convention
19. [ ] All group member names included
20. [ ] Ready to submit!

## 🚀 Submission Steps

1. **Final Testing**
   ```bash
   # Clean environment test
   cd /tmp
   unzip Group_X_Hybrid_RAG.zip
   cd Group_X_Hybrid_RAG
   python -m venv test_env
   source test_env/bin/activate
   pip install -r requirements.txt
   python setup.py  # Test setup
   ```

2. **Create ZIP**
   ```bash
   cd ConvAI_assingment_2
   cd ..
   zip -r Group_X_Hybrid_RAG.zip ConvAI_assingment_2/ \
       -x "*.git*" -x "*__pycache__*" -x "*venv*" -x "*.DS_Store"
   ```

3. **Verify ZIP**
   - Size: Should be < 200MB
   - Structure: Proper folder organization
   - Contents: All required files present

4. **Submit**
   - Upload to submission portal
   - Verify upload successful
   - Keep backup copy

## 📧 Submission Email Template (if required)

```
Subject: [ConvAI Assignment 2] Group X - Hybrid RAG System

Dear Professor,

Please find attached our submission for Assignment 2: Hybrid RAG System.

Group Members:
- [Name 1] - [ID]
- [Name 2] - [ID]
- [Name 3] - [ID]

Project Summary:
- Implemented hybrid RAG with dense (FAISS) + sparse (BM25) + RRF
- Generated 100 diverse evaluation questions
- Evaluated with MRR, BERTScore F1, and NDCG@5
- Includes ablation study, error analysis, and LLM-as-judge
- Complete Streamlit UI for interactive testing

File: Group_X_Hybrid_RAG.zip
Size: [X] MB

All code has been tested and runs successfully. Detailed setup and 
usage instructions are provided in README.md.

Thank you,
Group X
```

## ✅ Final Checklist

- [ ] Code complete and tested
- [ ] Documentation complete
- [ ] All metrics implemented with justifications
- [ ] Evaluation results generated
- [ ] Visualizations created
- [ ] UI screenshots taken
- [ ] ZIP file created
- [ ] ZIP file tested
- [ ] Submission ready
- [ ] **SUBMIT!** 🚀

---

**Good luck with your submission!** 🎉
