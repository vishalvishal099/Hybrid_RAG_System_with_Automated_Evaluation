#!/usr/bin/env python3
"""
Quick Test Script for Hybrid RAG System
This script performs a quick test of the system without full evaluation.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import HybridRAGSystem

def main():
    """Run quick test queries."""
    
    print("=" * 60)
    print("HYBRID RAG SYSTEM - QUICK TEST")
    print("=" * 60)
    print()
    
    # Initialize system
    print("Initializing system...")
    rag = HybridRAGSystem()
    
    # Load corpus
    if not os.path.exists("data/corpus.json"):
        print("Error: Corpus not found. Run data collection first:")
        print("  python src/data_collection.py")
        return
    
    print("Loading corpus...")
    rag.load_corpus()
    print(f"✓ Loaded {len(rag.corpus)} documents")
    print()
    
    # Build or load indexes
    if not os.path.exists("models/faiss_index"):
        print("Building indexes (this may take a few minutes)...")
        rag.build_dense_index()
        rag.build_sparse_index()
        print("✓ Indexes built")
    else:
        print("Loading existing indexes...")
        rag.load_dense_index()
        rag.load_sparse_index()
        print("✓ Indexes loaded")
    print()
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "Explain quantum computing",
        "Who was Albert Einstein?",
        "What causes climate change?",
        "How does photosynthesis work?"
    ]
    
    print("=" * 60)
    print("RUNNING TEST QUERIES")
    print("=" * 60)
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query}")
        print("-" * 60)
        
        try:
            # Run query
            result = rag.query(query, k=5)
            
            # Display results
            print(f"Answer: {result['answer'][:200]}...")
            print()
            print("Top 3 Sources:")
            for j, doc in enumerate(result['retrieved_docs'][:3], 1):
                print(f"  {j}. {doc['url']}")
                print(f"     Score: {doc['combined_score']:.4f}")
            print()
            
            # Display timing
            timing = result['timing']
            print(f"Timing:")
            print(f"  Dense: {timing['dense_retrieval']:.2f}s")
            print(f"  Sparse: {timing['sparse_retrieval']:.2f}s")
            print(f"  Generation: {timing['generation']:.2f}s")
            print(f"  Total: {timing['total']:.2f}s")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()
        print("=" * 60)
        print()
    
    print("TEST COMPLETE!")
    print()
    print("Next steps:")
    print("  1. Run full evaluation: python evaluation/pipeline.py")
    print("  2. Launch UI: streamlit run app.py")
    print()

if __name__ == "__main__":
    main()
