"""
Quick test to check if answer generation is working properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_system import HybridRAGSystem

# Initialize RAG
print("Loading RAG system...")
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()

# Test queries
test_questions = [
    "What is artificial intelligence?",
    "Who invented the telephone?",
    "When was the Roman Empire founded?"
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    
    response = rag.query(question, method="hybrid")
    
    print(f"\nAnswer: {response['answer']}")
    print(f"\nTop source: {response['sources'][0]['title']}")
    print(f"Preview: {response['sources'][0]['text_preview'][:200]}...")
    print(f"\nRetrieval time: {response['metadata']['retrieval_time']:.3f}s")
    print(f"Generation time: {response['metadata']['generation_time']:.3f}s")
