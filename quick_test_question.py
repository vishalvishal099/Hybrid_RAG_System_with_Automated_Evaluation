"""
Quick Test - Test a single question immediately
Usage: python quick_test_question.py "Your question here"
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import HybridRAGSystem

def main():
    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is artificial intelligence?"
    
    print(f"\n🔍 Testing Question: '{question}'\n")
    print("⏳ Loading RAG system...")
    
    # Initialize
    rag = HybridRAGSystem()
    rag.load_corpus()
    rag.load_indexes()
    
    print("✅ System loaded!\n")
    print("🤔 Generating answer...\n")
    
    # Query
    result = rag.query(question, method="hybrid")
    
    # Display results
    print("="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    print(f"\n✅ ANSWER:\n{result['answer']}\n")
    print("="*80)
    
    # Show top source
    if result.get('sources'):
        top_source = result['sources'][0]
        print(f"\n📚 Top Source: {top_source['title']}")
        print(f"🔗 URL: {top_source['url']}")
        print(f"📊 RRF Score: {top_source['scores']['rrf']:.4f}\n")
    
    # Show timing info if available
    if 'generation_time' in result:
        print(f"⏱️  Generation Time: {result['generation_time']:.2f}s")
    if 'retrieval_time' in result:
        print(f"🔍 Retrieval Time: {result['retrieval_time']:.2f}s")
    if 'generation_time' in result and 'retrieval_time' in result:
        print(f"⏰ Total Time: {result['retrieval_time'] + result['generation_time']:.2f}s")
    print()

if __name__ == "__main__":
    main()
