"""
Quick test of LLM-as-Judge implementation
Tests the judge on sample questions
"""

import json
import sys
sys.path.insert(0, '.')

from evaluation.metrics import RAGEvaluator


def test_llm_judge():
    """Test LLM judge with sample cases"""
    
    evaluator = RAGEvaluator()
    
    print("="*70)
    print("TESTING LLM-AS-JUDGE IMPLEMENTATION")
    print("="*70)
    
    # Test cases
    test_cases = [
        {
            "query": "What is machine learning?",
            "generated": "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed.",
            "ground_truth": "Machine learning is a type of AI that allows computers to learn and improve from experience without explicit programming.",
            "context": ["Machine learning is artificial intelligence", "ML systems learn from data"]
        },
        {
            "query": "Who invented the telephone?",
            "generated": "The telephone was invented by Alexander Graham Bell in 1876.",
            "ground_truth": "Alexander Graham Bell invented the telephone in 1876.",
            "context": ["Alexander Graham Bell invented telephone", "Bell's telephone patent 1876"]
        },
        {
            "query": "What is photosynthesis?",
            "generated": "The sky is blue and grass is green.",
            "ground_truth": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "context": ["Plants use photosynthesis", "Sunlight energy conversion"]
        }
    ]
    
    print("\n📝 Testing 3 sample cases:\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}] Question: {case['query']}")
        print(f"    Generated: {case['generated'][:80]}...")
        print(f"    Ground Truth: {case['ground_truth'][:80]}...")
        
        scores = evaluator.llm_judge_answer(
            query=case['query'],
            generated_answer=case['generated'],
            ground_truth=case['ground_truth'],
            retrieved_context=case['context']
        )
        
        print(f"\n    📊 Scores:")
        print(f"       Factual Accuracy:   {scores['factual_accuracy']:.3f}")
        print(f"       Completeness:       {scores['completeness']:.3f}")
        print(f"       Relevance:          {scores['relevance']:.3f}")
        print(f"       Coherence:          {scores['coherence']:.3f}")
        print(f"       No Hallucinations:  {scores['hallucination_score']:.3f}")
        print(f"       Overall:            {scores['overall']:.3f}")
        print(f"\n    💬 {scores['explanation']}")
        print("-"*70)
    
    print("\n✅ LLM-as-Judge test complete!")
    print("\nThe judge evaluates 5 dimensions:")
    print("  1. Factual Accuracy - matches ground truth (ROUGE-L)")
    print("  2. Completeness - covers key information")
    print("  3. Relevance - answers the query")
    print("  4. Coherence - well-structured text")
    print("  5. Hallucination - supported by context")


if __name__ == "__main__":
    test_llm_judge()
