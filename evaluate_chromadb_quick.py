"""
Quick Evaluation of ChromaDB Hybrid RAG System
Tests on 10 questions instead of 100 for faster results
"""

import json
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from chromadb_rag_system import ChromaDBHybridRAG


def load_evaluation_data(num_questions=10):
    """Load first N evaluation questions"""
    print(f"\n📂 Loading {num_questions} evaluation questions...")
    with open('data/questions_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data.get('questions', data)  # Handle both dict and list formats
    if isinstance(questions, dict):
        questions = list(questions.values())
    questions = questions[:num_questions]  # Take only first N questions
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


def calculate_f1_score(prediction, reference):
    """Calculate token-level F1 score"""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common_tokens = pred_tokens & ref_tokens
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def evaluate_system(rag_system, questions, method="hybrid"):
    """Evaluate RAG system on questions"""
    print(f"\n🔍 Evaluating with {method} retrieval...")
    
    results = []
    metrics = defaultdict(list)
    
    for q in tqdm(questions, desc=f"Evaluating {method}"):
        try:
            # Query the system
            start_time = time.time()
            result = rag_system.query(q['question'], method=method)
            query_time = time.time() - start_time
            
            # Calculate retrieval metrics
            mrr = calculate_mrr(result['sources'], q['url'])
            recall_at_10 = calculate_recall_at_k(result['sources'], q['url'], k=10)
            
            # Calculate answer quality metrics
            answer_f1 = calculate_f1_score(result['answer'], q.get('expected_answer', ''))
            
            # Store metrics
            metrics['MRR'].append(mrr)
            metrics['Recall@10'].append(recall_at_10)
            metrics['Answer_F1'].append(answer_f1)
            metrics['Total_Time'].append(query_time)
            
            # Store detailed results
            results.append({
                'question': q['question'],
                'expected_url': q['url'],
                'retrieved_top1_url': result['sources'][0].get('url') if result['sources'] else None,
                'answer': result['answer'],
                'mrr': mrr,
                'recall@10': recall_at_10,
                'answer_f1': answer_f1,
                'total_time': query_time,
                'method': method
            })
            
        except Exception as e:
            print(f"\n⚠️ Error processing question: {q['question'][:50]}...")
            print(f"   Error: {str(e)}")
            continue
    
    # Calculate aggregate metrics
    aggregate = {
        'method': method,
        'num_questions': len(results),
        'avg_mrr': sum(metrics['MRR']) / len(metrics['MRR']) if metrics['MRR'] else 0,
        'avg_recall@10': sum(metrics['Recall@10']) / len(metrics['Recall@10']) if metrics['Recall@10'] else 0,
        'avg_answer_f1': sum(metrics['Answer_F1']) / len(metrics['Answer_F1']) if metrics['Answer_F1'] else 0,
        'avg_total_time': sum(metrics['Total_Time']) / len(metrics['Total_Time']) if metrics['Total_Time'] else 0
    }
    
    return results, aggregate


def print_results(aggregate_results):
    """Print evaluation results"""
    print("\n" + "=" * 80)
    print("QUICK EVALUATION RESULTS")
    print("=" * 80)
    
    for result in aggregate_results:
        print(f"\n{'=' * 80}")
        print(f"Method: {result['method'].upper()}")
        print(f"{'=' * 80}")
        print(f"  Questions Evaluated: {result['num_questions']}")
        print(f"\n  Retrieval Metrics:")
        print(f"    Mean Reciprocal Rank (MRR):  {result['avg_mrr']:.4f}")
        print(f"    Recall@10:                    {result['avg_recall@10']:.4f}")
        print(f"\n  Answer Quality:")
        print(f"    Token F1:                     {result['avg_answer_f1']:.4f}")
        print(f"\n  Performance:")
        print(f"    Avg Query Time:               {result['avg_total_time']:.3f}s")


def main():
    """Main evaluation pipeline"""
    print("\n" + "=" * 80)
    print("CHROMADB HYBRID RAG - QUICK EVALUATION (10 questions)")
    print("=" * 80)
    
    # Load evaluation data (only 10 questions)
    questions = load_evaluation_data(num_questions=10)
    
    # Initialize RAG system
    print("\n🔧 Loading RAG system...")
    rag_system = ChromaDBHybridRAG()
    print("✓ RAG system loaded")
    
    # Run evaluations for all methods
    methods = ["dense", "sparse", "hybrid"]
    all_detailed_results = []
    aggregate_results = []
    
    for method in methods:
        detailed_results, aggregate = evaluate_system(rag_system, questions, method=method)
        all_detailed_results.extend(detailed_results)
        aggregate_results.append(aggregate)
    
    # Print results
    print_results(aggregate_results)
    
    # Save results
    df = pd.DataFrame(all_detailed_results)
    df.to_csv('evaluation_quick_chromadb.csv', index=False)
    
    with open('evaluation_quick_summary.json', 'w', encoding='utf-8') as f:
        json.dump(aggregate_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ QUICK EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📊 Results saved to:")
    print("  - evaluation_quick_chromadb.csv")
    print("  - evaluation_quick_summary.json")
    print("\n💡 Run 'python evaluate_chromadb.py' for full 100-question evaluation")
    print("\n")


if __name__ == "__main__":
    main()
