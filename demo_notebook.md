# Hybrid RAG System - Complete Demo Notebook

This notebook demonstrates the complete Hybrid RAG system implementation.

## Table of Contents
1. Setup and Imports
2. Data Collection
3. Index Building
4. Question Generation
5. RAG System Demo
6. Evaluation
7. Results Analysis

## 1. Setup and Imports

```python
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.data_collection import WikipediaDataCollector
from src.rag_system import HybridRAGSystem
from src.question_generation import QuestionGenerator
from evaluation.metrics import RAGEvaluator
from evaluation.innovative_eval import InnovativeEvaluator
from evaluation.pipeline import EvaluationPipeline

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
```

## 2. Data Collection

### Collect Wikipedia Articles

```python
# Initialize collector
collector = WikipediaDataCollector()

# Generate or load fixed URLs
print("Step 1: Collecting Wikipedia articles...")
corpus = collector.collect_and_process_corpus(use_existing_fixed=True)

print(f"\n✓ Corpus created:")
print(f"  Total URLs: {corpus['metadata']['total_urls']}")
print(f"  Total Chunks: {corpus['metadata']['total_chunks']}")
print(f"  Fixed URLs: {corpus['metadata']['fixed_urls']}")
print(f"  Random URLs: {corpus['metadata']['random_urls']}")
```

### Explore the Corpus

```python
# Load corpus
with open('data/corpus.json', 'r') as f:
    corpus_data = json.load(f)

# Show statistics
print(f"Documents: {len(corpus_data['documents'])}")
print(f"Chunks: {len(corpus_data['chunks'])}")

# Sample document
sample_doc = corpus_data['documents'][0]
print(f"\nSample Document:")
print(f"  Title: {sample_doc['title']}")
print(f"  URL: {sample_doc['url']}")
print(f"  Word Count: {sample_doc['word_count']}")
print(f"  Chunks: {sample_doc['num_chunks']}")

# Sample chunk
sample_chunk = corpus_data['chunks'][0]
print(f"\nSample Chunk:")
print(f"  Chunk ID: {sample_chunk['chunk_id']}")
print(f"  Title: {sample_chunk['title']}")
print(f"  Token Count: {sample_chunk['token_count']}")
print(f"  Text Preview: {sample_chunk['text'][:200]}...")
```

## 3. Index Building

### Build Dense Vector Index (FAISS)

```python
print("Step 2: Building indexes...")

# Initialize RAG system
rag = HybridRAGSystem()
rag.load_corpus()

# Build dense index
print("\nBuilding dense vector index...")
rag.build_dense_index()
print("✓ Dense index created")
```

### Build Sparse BM25 Index

```python
# Build sparse index
print("\nBuilding sparse BM25 index...")
rag.build_sparse_index()
print("✓ Sparse index created")
```

## 4. Question Generation

### Generate Evaluation Questions

```python
print("Step 3: Generating evaluation questions...")

# Initialize generator
generator = QuestionGenerator()
generator.load_corpus()

# Generate questions
questions = generator.generate_all_questions(num_total=100)

# Save questions
generator.save_questions(questions)

print(f"\n✓ Generated {len(questions)} questions")
```

### Analyze Question Distribution

```python
# Load questions
with open('data/questions_100.json', 'r') as f:
    q_data = json.load(f)

questions = q_data['questions']

# Count by type
from collections import Counter
type_counts = Counter([q['question_type'] for q in questions])
difficulty_counts = Counter([q['difficulty'] for q in questions])

# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Question types
axes[0].bar(type_counts.keys(), type_counts.values(), color='skyblue')
axes[0].set_title('Questions by Type', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Question Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Difficulty levels
axes[1].bar(difficulty_counts.keys(), difficulty_counts.values(), color='lightcoral')
axes[1].set_title('Questions by Difficulty', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Difficulty')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

print("\nQuestion Statistics:")
for q_type, count in type_counts.items():
    print(f"  {q_type}: {count}")
```

### Show Sample Questions

```python
# Display sample questions
print("\n📋 Sample Questions:\n")
for i, q in enumerate(questions[:5], 1):
    print(f"{i}. Type: {q['question_type']} | Difficulty: {q['difficulty']}")
    print(f"   Q: {q['question']}")
    print(f"   Source: {q['source_title']}")
    print()
```

## 5. RAG System Demo

### Test Single Query

```python
print("Step 4: Testing RAG system...")

# Load RAG system
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()

# Test query
test_query = "What is artificial intelligence?"
print(f"\n🔍 Query: {test_query}\n")

# Get response
response = rag.query(test_query, method="hybrid")

print(f"✨ Answer:\n{response['answer']}\n")

print(f"📚 Sources:")
for i, source in enumerate(response['sources'], 1):
    print(f"{i}. {source['title']}")
    print(f"   Dense: {source['scores']['dense']:.4f} | "
          f"Sparse: {source['scores']['sparse']:.4f} | "
          f"RRF: {source['scores']['rrf']:.4f}")
    print(f"   Preview: {source['text_preview']}")
    print()

print(f"⏱️ Performance:")
print(f"   Retrieval: {response['metadata']['retrieval_time']:.3f}s")
print(f"   Generation: {response['metadata']['generation_time']:.3f}s")
print(f"   Total: {response['metadata']['total_time']:.3f}s")
```

### Compare Retrieval Methods

```python
# Test all three methods
methods = ['dense', 'sparse', 'hybrid']
test_query = "Who invented the telephone?"

print(f"\n🔬 Comparing Retrieval Methods")
print(f"Query: {test_query}\n")

results = {}
for method in methods:
    response = rag.query(test_query, method=method)
    results[method] = response
    
    print(f"\n{method.upper()} Method:")
    print(f"  Answer: {response['answer'][:100]}...")
    print(f"  Top Source: {response['sources'][0]['title']}")
    print(f"  Time: {response['metadata']['total_time']:.3f}s")
```

## 6. Evaluation

### Run Full Evaluation

```python
print("Step 5: Running evaluation...")

# Initialize evaluator
evaluator = RAGEvaluator()

# Evaluate subset of questions
eval_results = []

for q in questions[:20]:  # Evaluate first 20 for demo
    response = rag.query(q['question'], method='hybrid')
    eval_result = evaluator.evaluate_single_query(q, response)
    eval_results.append(eval_result)

# Aggregate results
aggregated = evaluator.aggregate_results(eval_results)

print("\n📊 Evaluation Results:")
print(f"  MRR: {aggregated['overall_metrics']['mrr']:.4f}")
print(f"  NDCG@5: {aggregated['overall_metrics']['ndcg_5']:.4f}")
print(f"  BERTScore F1: {aggregated['overall_metrics']['bertscore_f1']:.4f}")
print(f"  Precision@5: {aggregated['overall_metrics']['precision_5']:.4f}")
print(f"  Recall@5: {aggregated['overall_metrics']['recall_5']:.4f}")
```

### Ablation Study

```python
print("\n🔬 Ablation Study: Comparing Methods...")

innovative_eval = InnovativeEvaluator()
ablation_results = innovative_eval.ablation_study(rag, questions[:20])

# Plot comparison
methods = list(ablation_results.keys())
mrr_scores = [ablation_results[m]['overall_metrics']['mrr'] for m in methods]
ndcg_scores = [ablation_results[m]['overall_metrics']['ndcg_5'] for m in methods]
bert_scores = [ablation_results[m]['overall_metrics']['bertscore_f1'] for m in methods]

x = range(len(methods))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar([i - width for i in x], mrr_scores, width, label='MRR', color='#3498db')
plt.bar(x, ndcg_scores, width, label='NDCG@5', color='#2ecc71')
plt.bar([i + width for i in x], bert_scores, width, label='BERTScore F1', color='#f39c12')

plt.xlabel('Method', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Ablation Study: Method Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, [m.upper() for m in methods])
plt.legend()
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
```

## 7. Results Analysis

### Performance by Question Type

```python
# Analyze by question type
by_type = aggregated['by_question_type']

# Create DataFrame
type_df = pd.DataFrame([
    {
        'Type': q_type,
        'Count': data['count'],
        'MRR': data['mrr'],
        'NDCG@5': data['ndcg_5'],
        'BERTScore F1': data['bertscore_f1']
    }
    for q_type, data in by_type.items()
])

print("\n📊 Performance by Question Type:")
print(type_df.to_string(index=False))

# Visualize
type_df.set_index('Type')[['MRR', 'NDCG@5', 'BERTScore F1']].plot(kind='bar', figsize=(12, 6))
plt.title('Performance by Question Type', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xlabel('Question Type')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()
```

### Error Analysis

```python
# Categorize results
successes = [r for r in eval_results if r['metrics']['mrr'] > 0.5 and r['metrics']['bertscore_f1'] > 0.7]
partial = [r for r in eval_results if 0 < r['metrics']['mrr'] <= 0.5 or 0.5 <= r['metrics']['bertscore_f1'] <= 0.7]
failures = [r for r in eval_results if r['metrics']['mrr'] == 0 or r['metrics']['bertscore_f1'] < 0.5]

print(f"\n🎯 Performance Categories:")
print(f"  Success: {len(successes)} ({len(successes)/len(eval_results)*100:.1f}%)")
print(f"  Partial Success: {len(partial)} ({len(partial)/len(eval_results)*100:.1f}%)")
print(f"  Failures: {len(failures)} ({len(failures)/len(eval_results)*100:.1f}%)")

# Show failure examples
if failures:
    print(f"\n❌ Failure Examples:")
    for i, fail in enumerate(failures[:3], 1):
        print(f"\n{i}. Question: {fail['question']}")
        print(f"   Type: {fail['question_type']}")
        print(f"   MRR: {fail['metrics']['mrr']:.4f}")
        print(f"   BERTScore: {fail['metrics']['bertscore_f1']:.4f}")
```

### Time Analysis

```python
# Analyze response times
retrieval_times = [r['retrieval_time'] for r in eval_results]
generation_times = [r['generation_time'] for r in eval_results]
total_times = [r['total_time'] for r in eval_results]

print(f"\n⏱️ Time Statistics:")
print(f"  Avg Retrieval: {np.mean(retrieval_times):.3f}s")
print(f"  Avg Generation: {np.mean(generation_times):.3f}s")
print(f"  Avg Total: {np.mean(total_times):.3f}s")

# Plot time distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(retrieval_times, bins=15, color='skyblue', edgecolor='black')
axes[0].set_title('Retrieval Time Distribution')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Frequency')

axes[1].hist(generation_times, bins=15, color='lightgreen', edgecolor='black')
axes[1].set_title('Generation Time Distribution')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Frequency')

axes[2].hist(total_times, bins=15, color='lightcoral', edgecolor='black')
axes[2].set_title('Total Time Distribution')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## Summary

```python
print("\n" + "="*60)
print("HYBRID RAG SYSTEM - DEMO COMPLETE")
print("="*60)
print("\nKey Achievements:")
print("✓ Collected and processed 500 Wikipedia articles")
print("✓ Built dense (FAISS) and sparse (BM25) indexes")
print("✓ Generated 100 diverse evaluation questions")
print("✓ Implemented hybrid retrieval with RRF")
print("✓ Evaluated with MRR, NDCG@5, and BERTScore")
print("✓ Performed ablation study and error analysis")
print("\nNext Steps:")
print("- Run full evaluation: python evaluation/pipeline.py")
print("- Launch UI: streamlit run app.py")
print("="*60)
```

## Notes

- This notebook demonstrates the core functionality
- For full evaluation, use the automated pipeline: `python evaluation/pipeline.py`
- The Streamlit UI provides an interactive interface
- All results are saved to the `reports/` directory
- Adjust `config.yaml` for different settings

---

**Built for Educational Purposes** | Conversational AI Assignment
