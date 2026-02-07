"""
Comprehensive Evaluation of ChromaDB Hybrid RAG System
Evaluates retrieval and generation performance on 100 questions
"""

import json
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from bert_score import score as bert_score
from chromadb_rag_system import ChromaDBHybridRAG


def load_evaluation_data():
    """Load 100 evaluation questions"""
    print("\n📂 Loading evaluation data...")
    with open('data/questions_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data.get('questions', data)  # Handle both dict and list formats
    if isinstance(questions, dict):
        questions = list(questions.values())
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
    """Evaluate RAG system on all questions"""
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
            metrics['Retrieval_Time'].append(result['retrieval_time'])
            metrics['Generation_Time'].append(result['generation_time'])
            metrics['Total_Time'].append(query_time)
            
            # Store detailed results
            results.append({
                'question': q['question'],
                'expected_url': q['url'],
                'retrieved_top1_url': result['sources'][0].get('url') if result['sources'] else None,
                'answer': result['answer'],
                'expected_answer': q.get('expected_answer', ''),
                'mrr': mrr,
                'recall@10': recall_at_10,
                'answer_f1': answer_f1,
                'retrieval_time': result['retrieval_time'],
                'generation_time': result['generation_time'],
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
        'avg_retrieval_time': sum(metrics['Retrieval_Time']) / len(metrics['Retrieval_Time']) if metrics['Retrieval_Time'] else 0,
        'avg_generation_time': sum(metrics['Generation_Time']) / len(metrics['Generation_Time']) if metrics['Generation_Time'] else 0,
        'avg_total_time': sum(metrics['Total_Time']) / len(metrics['Total_Time']) if metrics['Total_Time'] else 0
    }
    
    return results, aggregate


def calculate_bert_score(predictions, references):
    """Calculate BERTScore for answer quality"""
    print("\n📊 Calculating BERTScore...")
    
    # Filter out empty predictions/references
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    
    if not valid_pairs:
        print("⚠️ No valid prediction-reference pairs for BERTScore")
        return None
    
    preds, refs = zip(*valid_pairs)
    
    # Calculate BERTScore
    P, R, F1 = bert_score(
        list(preds),
        list(refs),
        lang='en',
        verbose=False,
        device='cpu'
    )
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


def print_results(aggregate_results):
    """Print evaluation results"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for result in aggregate_results:
        print(f"\n{'=' * 80}")
        print(f"Method: {result['method'].upper()}")
        print(f"{'=' * 80}")
        print(f"  Questions Evaluated: {result['num_questions']}")
        print(f"\n  Retrieval Metrics:")
        print(f"    Mean Reciprocal Rank (MRR):  {result['avg_mrr']:.4f}")
        print(f"    Recall@10:                    {result['avg_recall@10']:.4f}")
        print(f"\n  Answer Quality Metrics:")
        print(f"    Token F1:                     {result['avg_answer_f1']:.4f}")
        if result.get('bert_score'):
            print(f"    BERTScore F1:                 {result['bert_score']['f1']:.4f}")
            print(f"    BERTScore Precision:          {result['bert_score']['precision']:.4f}")
            print(f"    BERTScore Recall:             {result['bert_score']['recall']:.4f}")
        print(f"\n  Performance Metrics:")
        print(f"    Avg Retrieval Time:           {result['avg_retrieval_time']:.3f}s")
        print(f"    Avg Generation Time:          {result['avg_generation_time']:.3f}s")
        print(f"    Avg Total Time:               {result['avg_total_time']:.3f}s")


def save_results(detailed_results, aggregate_results):
    """Save results to files"""
    print("\n💾 Saving results...")
    
    # Save detailed results to CSV
    df = pd.DataFrame(detailed_results)
    df.to_csv('evaluation_results_chromadb.csv', index=False)
    print("✓ Saved detailed results to: evaluation_results_chromadb.csv")
    
    # Save aggregate results to JSON
    with open('evaluation_summary_chromadb.json', 'w', encoding='utf-8') as f:
        json.dump(aggregate_results, f, indent=2)
    print("✓ Saved aggregate results to: evaluation_summary_chromadb.json")
    
    # Create comparison table
    comparison_df = pd.DataFrame(aggregate_results)
    comparison_df.to_csv('evaluation_comparison_chromadb.csv', index=False)
    print("✓ Saved comparison table to: evaluation_comparison_chromadb.csv")


def main():
    """Main evaluation pipeline"""
    print("\n" + "=" * 80)
    print("CHROMADB HYBRID RAG SYSTEM - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Load evaluation data
    questions = load_evaluation_data()
    
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
        
        # Calculate BERTScore for answers
        predictions = [r['answer'] for r in detailed_results]
        references = [r['expected_answer'] for r in detailed_results if r.get('expected_answer')]
        
        if references and len(references) == len(predictions):
            bert_scores = calculate_bert_score(predictions, references)
            aggregate['bert_score'] = bert_scores
        
        all_detailed_results.extend(detailed_results)
        aggregate_results.append(aggregate)
    
    # Print results
    print_results(aggregate_results)
    
    # Save results
    save_results(all_detailed_results, aggregate_results)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📊 Results saved to:")
    print("  - evaluation_results_chromadb.csv (detailed per-question results)")
    print("  - evaluation_summary_chromadb.json (aggregate metrics)")
    print("  - evaluation_comparison_chromadb.csv (method comparison)")
    print("\n")


if __name__ == "__main__":
    main()
