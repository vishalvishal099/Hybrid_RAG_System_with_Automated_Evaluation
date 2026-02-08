"""
Run LLM-as-Judge Evaluation
Evaluates answer quality on 5 dimensions without needing API keys
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chromadb_rag_system import ChromaDBHybridRAG
from evaluation.run_evaluation import ComprehensiveEvaluationPipeline


def main():
    print("="*70)
    print("LLM-AS-JUDGE EVALUATION - HEURISTIC-BASED (NO API REQUIRED)")
    print("="*70)
    
    # Initialize RAG system
    print("\n🔧 Initializing ChromaDB Hybrid RAG System...")
    rag_system = ChromaDBHybridRAG(
        collection_name="wikipedia_articles",
        persist_directory="chroma_db"
    )
    print(f"✓ System ready with {len(rag_system.corpus)} chunks")
    
    # Load test questions
    print("\n📥 Loading test questions...")
    questions_file = Path("data/questions_100.json")
    
    if not questions_file.exists():
        print(f"❌ Error: {questions_file} not found!")
        return
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"✓ Loaded {len(questions)} questions")
    
    # Initialize evaluation pipeline
    pipeline = ComprehensiveEvaluationPipeline(
        rag_system=rag_system,
        output_dir="evaluation/llm_judge_results"
    )
    
    # Run LLM-judge evaluation
    print("\n🤖 Running LLM-as-Judge Evaluation...")
    print("This evaluates 5 dimensions:")
    print("  1. Factual Accuracy - Correctness vs ground truth")
    print("  2. Completeness - Coverage of key information")
    print("  3. Relevance - Answers the actual question")
    print("  4. Coherence - Well-structured and readable")
    print("  5. Hallucination - No unsupported claims")
    print()
    
    results = pipeline.run_llm_judge_evaluation(
        test_cases=questions,
        sample_size=50  # Evaluate 50 questions
    )
    
    # Save results
    output_file = Path("evaluation/llm_judge_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*70)
    print("📊 LLM-AS-JUDGE SUMMARY")
    print("="*70)
    print(f"Questions Evaluated: {results['sample_size']}")
    print(f"\nScores (0.0 - 1.0, higher is better):")
    print(f"  Factual Accuracy:   {results['mean_factual_accuracy']:.4f}")
    print(f"  Completeness:       {results['mean_completeness']:.4f}")
    print(f"  Relevance:          {results['mean_relevance']:.4f}")
    print(f"  Coherence:          {results['mean_coherence']:.4f}")
    print(f"  No Hallucinations:  {results['mean_hallucination_score']:.4f}")
    print(f"  {'─'*50}")
    print(f"  Overall Score:      {results['mean_overall']:.4f}")
    
    # Interpretation
    overall = results['mean_overall']
    if overall >= 0.8:
        grade = "🏆 EXCELLENT"
    elif overall >= 0.7:
        grade = "✅ GOOD"
    elif overall >= 0.6:
        grade = "⚠️  MODERATE"
    else:
        grade = "❌ NEEDS IMPROVEMENT"
    
    print(f"\n  Quality Rating: {grade}")
    print("="*70)
    
    # Show sample evaluations
    if results.get('detailed_results'):
        print("\n📝 Sample Evaluations (First 3):")
        for i, detail in enumerate(results['detailed_results'][:3], 1):
            print(f"\n  [{i}] {detail['question'][:60]}...")
            print(f"      Generated: {detail['generated_answer'][:80]}...")
            print(f"      {detail['scores']['explanation']}")
    
    print("\n✅ LLM-as-Judge evaluation complete!")


if __name__ == "__main__":
    main()
