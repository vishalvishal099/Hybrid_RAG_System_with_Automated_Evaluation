# Top 20 Questions with Perfect Retrieval Match

**System Status:** ✅ All servers running
- **Streamlit UI:** http://localhost:8501
- **Network URL:** http://192.168.29.144:8501

These 20 questions achieved **perfect retrieval performance** (MRR = 1.0, Recall@10 = 1.0), meaning the RAG system retrieved the exact correct Wikipedia article as the top result.

## Questions with Perfect Match

### 1. When did Niagara Falls happen?
- **Expected:** https://en.wikipedia.org/wiki/Niagara_Falls
- **Retrieved:** https://en.wikipedia.org/wiki/Niagara_Falls ✅
- **Method:** Dense (ChromaDB)

### 2. Where is HEAnet located?
- **Expected:** https://en.wikipedia.org/wiki/HEAnet
- **Retrieved:** https://en.wikipedia.org/wiki/HEAnet ✅
- **Method:** Dense (ChromaDB)

### 3. What is Rock music?
- **Expected:** https://en.wikipedia.org/wiki/Rock_music
- **Retrieved:** https://en.wikipedia.org/wiki/Rock_music ✅
- **Method:** Dense (ChromaDB)

### 4. How is Vincent van Gogh used?
- **Expected:** https://en.wikipedia.org/wiki/Vincent_van_Gogh
- **Retrieved:** https://en.wikipedia.org/wiki/Vincent_van_Gogh ✅
- **Method:** Dense (ChromaDB)

### 5. Why was Same-sex marriage in Germany created?
- **Expected:** https://en.wikipedia.org/wiki/Same-sex_marriage_in_Germany
- **Retrieved:** https://en.wikipedia.org/wiki/Same-sex_marriage_in_Germany ✅
- **Method:** Dense (ChromaDB)

### 6. When was Bokaro Steel City established?
- **Expected:** https://en.wikipedia.org/wiki/Bokaro_Steel_City
- **Retrieved:** https://en.wikipedia.org/wiki/Bokaro_Steel_City ✅
- **Method:** Dense (ChromaDB)

### 7. How is Redox used?
- **Expected:** https://en.wikipedia.org/wiki/Redox
- **Retrieved:** https://en.wikipedia.org/wiki/Redox ✅
- **Method:** Dense (ChromaDB)

### 8. What is This attack was?
- **Expected:** https://en.wikipedia.org/wiki/Sockstress
- **Retrieved:** https://en.wikipedia.org/wiki/Sockstress ✅
- **Method:** Dense (ChromaDB)

### 9. When did Guards Armoured Division happen?
- **Expected:** https://en.wikipedia.org/wiki/Guards_Armoured_Division
- **Retrieved:** https://en.wikipedia.org/wiki/Guards_Armoured_Division ✅
- **Method:** Dense (ChromaDB)

### 10. What is Glycerate 3-phosphate, in?
- **Expected:** https://en.wikipedia.org/wiki/Photosynthesis
- **Retrieved:** https://en.wikipedia.org/wiki/Photosynthesis ✅
- **Method:** Dense (ChromaDB)

### 11. How is SS Cayuga used?
- **Expected:** https://en.wikipedia.org/wiki/SS_Cayuga
- **Retrieved:** https://en.wikipedia.org/wiki/SS_Cayuga ✅
- **Method:** Dense (ChromaDB)

### 12. How does The Colony (professional wrestling) work?
- **Expected:** https://en.wikipedia.org/wiki/The_Colony_(professional_wrestling)
- **Retrieved:** https://en.wikipedia.org/wiki/The_Colony_(professional_wrestling) ✅
- **Method:** Dense (ChromaDB)

### 13. What is The EU offers?
- **Expected:** https://en.wikipedia.org/wiki/European_Union
- **Retrieved:** https://en.wikipedia.org/wiki/European_Union ✅
- **Method:** Dense (ChromaDB)

### 14. Where is Data structure located?
- **Expected:** https://en.wikipedia.org/wiki/Data_structure
- **Retrieved:** https://en.wikipedia.org/wiki/Data_structure ✅
- **Method:** Dense (ChromaDB)

### 15. Why is Piano important?
- **Expected:** https://en.wikipedia.org/wiki/Piano
- **Retrieved:** https://en.wikipedia.org/wiki/Piano ✅
- **Method:** Dense (ChromaDB)

### 16. Where did Green and golden bell frog originate?
- **Expected:** https://en.wikipedia.org/wiki/Green_and_golden_bell_frog
- **Retrieved:** https://en.wikipedia.org/wiki/Green_and_golden_bell_frog ✅
- **Method:** Dense (ChromaDB)

### 17. Where did Pablo Picasso originate?
- **Expected:** https://en.wikipedia.org/wiki/Pablo_Picasso
- **Retrieved:** https://en.wikipedia.org/wiki/Pablo_Picasso ✅
- **Method:** Dense (ChromaDB)

### 18. What are the main features of The Shawshank Redemption?
- **Expected:** https://en.wikipedia.org/wiki/The_Shawshank_Redemption
- **Retrieved:** https://en.wikipedia.org/wiki/The_Shawshank_Redemption ✅
- **Method:** Dense (ChromaDB)

### 19. How does Linguistic discrimination work?
- **Expected:** https://en.wikipedia.org/wiki/Linguistic_discrimination
- **Retrieved:** https://en.wikipedia.org/wiki/Linguistic_discrimination ✅
- **Method:** Dense (ChromaDB)

### 20. When did HD 5388 happen?
- **Expected:** https://en.wikipedia.org/wiki/HD_5388
- **Retrieved:** https://en.wikipedia.org/wiki/HD_5388 ✅
- **Method:** Dense (ChromaDB)

---

## System Performance Summary

All 20 questions achieved:
- ✅ **MRR = 1.0** (Perfect ranking - correct document at position 1)
- ✅ **Recall@10 = 1.0** (Correct document found in top 10)
- ✅ **100% Complete Answers** (All answers were fully generated)

## Try These Questions in the UI

You can now test these questions in the Streamlit UI at:
- **Local:** http://localhost:8501
- **Network:** http://192.168.29.144:8501

Select any retrieval method (Dense, Sparse, or Hybrid) and try these questions to see the high-quality results!

## Notes

- All 20 questions used **Dense retrieval (ChromaDB)** method
- These represent the best performing questions from the full evaluation of 100 questions
- The system correctly identified and ranked the target Wikipedia article as the #1 result for all these queries
