"""
Quick test of confidence calibration with 5 questions
"""

import json
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './submission/01_source_code')

from chromadb_rag_system import ChromaDBHybridRAG


def test_confidence():
    print("="*70)
    print("TESTING CONFIDENCE SCORING")
    print("="*70)
    
    # Initialize
    print("\n🔧 Initializing RAG system...")
    rag_system = ChromaDBHybridRAG(chroma_path="chroma_db")
    
    # Test questions
    test_questions = [
        "What is artificial intelligence?",
        "Who invented the telephone?",
        "What is photosynthesis?"
    ]
    
    print("\n📝 Testing confidence scoring on 3 questions...\n")
    
    for i, q in enumerate(test_questions, 1):
        print(f"[{i}] {q}")
        
        try:
            result = rag_system.query(q, method="hybrid")
            
            answer = result.get('answer', 'No answer')
            confidence = result.get('confidence', 0.0)
            
            print(f"    Answer: {answer[:80]}...")
            print(f"    Confidence: {confidence:.4f}")
            print()
            
        except Exception as e:
            print(f"    ❌ Error: {e}\n")
    
    print("✅ Confidence scoring test complete!")


if __name__ == "__main__":
    test_confidence()
