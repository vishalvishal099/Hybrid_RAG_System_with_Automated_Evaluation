"""
Extended Ablation Study for Hybrid RAG System
Tests different K values, N values, and RRF k parameters
"""

import json
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from chromadb_rag_system import ChromaDBHybridRAG


def load_questions(limit=20):
    """Load subset of questions for faster ablation"""
    with open('data/questions_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions'][:limit]


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


def test_k_values(rag_system, questions):
    """Test different K values for retrieval"""
    print("\n🔬 Testing Different K Values (number of chunks retrieved)...")
    k_values = [5, 10, 15, 20]
    results = []
    
    for k in k_values:
        print(f"\n  Testing K={k}...")
        metrics = defaultdict(list)
        
        for q in tqdm(questions, desc=f"K={k}"):
            try:
                # Modify retrieval to use different K
                result = rag_system.query(q['question'], method='hybrid', top_k=k)
                
                mrr = calculate_mrr(result['sources'], q['source_url'])
                recall = calculate_recall_at_k(result['sources'], q['source_url'], k=k)
                
                metrics['MRR'].append(mrr)
                metrics['Recall'].append(recall)
                metrics['Retrieval_Time'].append(result['retrieval_time'])
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                continue
        
        results.append({
            'K': k,
            'Avg_MRR': sum(metrics['MRR']) / len(metrics['MRR']) if metrics['MRR'] else 0,
            'Avg_Recall': sum(metrics['Recall']) / len(metrics['Recall']) if metrics['Recall'] else 0,
            'Avg_Time': sum(metrics['Retrieval_Time']) / len(metrics['Retrieval_Time']) if metrics['Retrieval_Time'] else 0,
            'Num_Questions': len(metrics['MRR'])
        })
    
    return results


def test_rrf_k_values(rag_system, questions):
    """Test different RRF k values"""
    print("\n🔬 Testing Different RRF k Values...")
    rrf_k_values = [30, 60, 100]
    results = []
    
    for rrf_k in rrf_k_values:
        print(f"\n  Testing RRF k={rrf_k}...")
        metrics = defaultdict(list)
        
        # Temporarily modify RRF k
        original_k = rag_system.rrf_k
        rag_system.rrf_k = rrf_k
        
        for q in tqdm(questions, desc=f"RRF k={rrf_k}"):
            try:
                result = rag_system.query(q['question'], method='hybrid')
                
                mrr = calculate_mrr(result['sources'], q['source_url'])
                recall = calculate_recall_at_k(result['sources'], q['source_url'], k=10)
                
                metrics['MRR'].append(mrr)
                metrics['Recall@10'].append(recall)
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                continue
        
        results.append({
            'RRF_k': rrf_k,
            'Avg_MRR': sum(metrics['MRR']) / len(metrics['MRR']) if metrics['MRR'] else 0,
            'Avg_Recall@10': sum(metrics['Recall@10']) / len(metrics['Recall@10']) if metrics['Recall@10'] else 0,
            'Num_Questions': len(metrics['MRR'])
        })
        
        # Restore original k
        rag_system.rrf_k = original_k
    
    return results


def test_top_n_chunks(questions, limit=20):
    """Test different N values for answer generation (chunks used by LLM)"""
    print("\n🔬 Testing Different N Values (chunks for answer generation)...")
    print("Note: This tests how many retrieved chunks are used for answer generation")
    
    n_values = [3, 5, 7, 10]
    results = []
    
    for n in n_values:
        print(f"\n  Testing N={n} chunks for generation...")
        # This would require modifying the LLM prompt
        # For now, we'll simulate the impact on answer quality
        
        results.append({
            'N': n,
            'Description': f'Use top-{n} chunks for answer generation',
            'Expected_Impact': 'More chunks = more context but potentially more noise',
            'Status': 'Simulated (requires LLM prompt modification)'
        })
    
    return results


def main():
    """Run extended ablation study"""
    print("=" * 80)
    print("EXTENDED ABLATION STUDY")
    print("=" * 80)
    print("\nThis study tests the impact of different hyperparameters:")
    print("  - K: Number of chunks retrieved")
    print("  - RRF k: Rank fusion constant")
    print("  - N: Number of chunks used for generation")
    
    # Initialize RAG system
    print("\n🚀 Initializing ChromaDB Hybrid RAG System...")
    rag_system = ChromaDBHybridRAG()
    
    # Load questions (use subset for faster testing)
    questions = load_questions(limit=20)
    print(f"✓ Loaded {len(questions)} questions for ablation study")
    
    # Test 1: Different K values
    k_results = test_k_values(rag_system, questions)
    
    # Test 2: Different RRF k values
    rrf_results = test_rrf_k_values(rag_system, questions)
    
    # Test 3: Different N values (simulated)
    n_results = test_top_n_chunks(questions)
    
    # Save results
    output = {
        'study_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_questions_tested': len(questions),
        'k_value_results': k_results,
        'rrf_k_results': rrf_results,
        'n_value_results': n_results
    }
    
    with open('evaluation/ablation_study_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print results
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    
    print("\n📊 K Value Results (Number of chunks retrieved):")
    print("-" * 80)
    df_k = pd.DataFrame(k_results)
    print(df_k.to_string(index=False))
    
    print("\n📊 RRF k Results (Rank fusion constant):")
    print("-" * 80)
    df_rrf = pd.DataFrame(rrf_results)
    print(df_rrf.to_string(index=False))
    
    print("\n📊 N Value Results (Chunks for generation):")
    print("-" * 80)
    df_n = pd.DataFrame(n_results)
    print(df_n.to_string(index=False))
    
    print(f"\n✅ Results saved to evaluation/ablation_study_results.json")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    best_k = max(k_results, key=lambda x: x['Avg_MRR'])
    print(f"\n✓ Best K value: {best_k['K']} (MRR: {best_k['Avg_MRR']:.4f})")
    
    best_rrf = max(rrf_results, key=lambda x: x['Avg_MRR'])
    print(f"✓ Best RRF k value: {best_rrf['RRF_k']} (MRR: {best_rrf['Avg_MRR']:.4f})")
    
    print("\n✓ N value testing shows theoretical impact (requires implementation)")


if __name__ == "__main__":
    main()
