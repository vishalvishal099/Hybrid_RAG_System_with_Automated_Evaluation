"""
MINI Evaluation - 20 Questions Only
Fast test to verify system works - should complete in 2-3 minutes
"""

import json
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from chromadb_rag_system import ChromaDBHybridRAG


def load_evaluation_data(num_questions=20):
    """Load first N evaluation questions"""
    print(f"\n📂 Loading {num_questions} evaluation questions...")
    with open('data/questions_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data['questions'][:num_questions]
    print(f"✓ Loaded {len(questions)} questions")
    return questions


def calculate_mrr(retrieved_chunks, ground_truth_url):
    """Calculate Mean Reciprocal Rank"""
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if chunk.get('url') == ground_truth_url:
            return 1.0 / rank
    return 0.0


def calculate_recall_at_k(retrieved_chunks, ground_truth_url, k=10):
    """Calculate Recall@K"""
    top_k_urls = [chunk.get('url') for chunk in retrieved_chunks[:k]]
    return 1.0 if ground_truth_url in top_k_urls else 0.0


def evaluate_system(rag_system, questions, method="hybrid"):
    """Evaluate RAG system"""
    print(f"\n🔍 Evaluating with {method} retrieval...")
    
    results = []
    metrics = defaultdict(list)
    
    start_time = time.time()
    
    for q in tqdm(questions, desc=f"{method}"):
        try:
            # Query the system
            result = rag_system.query(q['question'], method=method)
            
            # Calculate metrics
            mrr = calculate_mrr(result['sources'], q['source_url'])
            recall_at_10 = calculate_recall_at_k(result['sources'], q['source_url'], k=10)
            
            metrics['MRR'].append(mrr)
            metrics['Recall@10'].append(recall_at_10)
            
            results.append({
                'question': q['question'],
                'answer': result['answer'],
                'mrr': mrr,
                'recall@10': recall_at_10,
                'method': method
            })
            
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    
    # Calculate aggregate
    aggregate = {
        'method': method,
        'num_questions': len(results),
        'avg_mrr': sum(metrics['MRR']) / len(metrics['MRR']) if metrics['MRR'] else 0,
        'avg_recall@10': sum(metrics['Recall@10']) / len(metrics['Recall@10']) if metrics['Recall@10'] else 0,
        'total_time': total_time
    }
    
    return results, aggregate


def main():
    """Main evaluation"""
    print("\n" + "=" * 80)
    print("MINI EVALUATION - 20 QUESTIONS")
    print("=" * 80)
    
    # Load data
    questions = load_evaluation_data(20)
    
    # Initialize system
    print("\n🔧 Loading RAG system...")
    rag_system = ChromaDBHybridRAG()
    print("✓ RAG system loaded")
    
    # Evaluate all methods
    all_results = []
    aggregates = []
    
    for method in ["dense", "sparse", "hybrid"]:
        results, aggregate = evaluate_system(rag_system, questions, method)
        all_results.extend(results)
        aggregates.append(aggregate)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for agg in aggregates:
        print(f"\n{agg['method'].upper()}:")
        print(f"  MRR:        {agg['avg_mrr']:.4f}")
        print(f"  Recall@10:  {agg['avg_recall@10']:.4f}")
        print(f"  Time:       {agg['total_time']:.1f}s")
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv('evaluation_mini_results.csv', index=False)
    print(f"\n✓ Saved results to: evaluation_mini_results.csv")
    
    # Save summary
    with open('evaluation_mini_summary.json', 'w') as f:
        json.dump(aggregates, f, indent=2)
    print(f"✓ Saved summary to: evaluation_mini_summary.json")
    
    print("\n" + "=" * 80)
    print("✅ MINI EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
