"""
Backend Testing Script - Test RAG System Directly
Run this to test questions without the UI
"""

import sys
sys.path.append('src')

from rag_system import HybridRAGSystem
import json
from pprint import pprint

def test_question(rag, question, show_sources=True):
    """Test a single question and show results"""
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    
    # Query the system
    result = rag.query(question, method="hybrid")
    
    # Display answer
    print(f"\n✅ ANSWER:")
    print(f"{result['answer']}")
    
    # Display generation time
    print(f"\n⏱️  Generation Time: {result['generation_time']:.2f}s")
    
    if show_sources and 'sources' in result:
        print(f"\n📚 TOP SOURCES:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"\n  Source {i}:")
            print(f"  📄 Title: {source['title']}")
            print(f"  🔗 URL: {source['url']}")
            print(f"  📊 RRF Score: {source['scores']['rrf']:.4f}")
            print(f"  📝 Preview: {source['text_preview'][:150]}...")
    
    return result


def main():
    print("\n🚀 Initializing RAG System...")
    
    # Initialize system
    rag = HybridRAGSystem()
    rag.load_corpus()
    rag.load_indexes()
    
    print("✅ System loaded successfully!\n")
    
    # Test questions from evaluation dataset
    test_questions = [
        "What is artificial intelligence?",
        "Who invented the telephone?",
        "When was the Roman Empire founded?",
        "What is DNA?",
        "Who painted the Mona Lisa?"
    ]
    
    # Interactive mode or batch mode
    if len(sys.argv) > 1:
        # Single question from command line
        question = " ".join(sys.argv[1:])
        test_question(rag, question)
    else:
        # Batch test all questions
        print("Testing all evaluation questions...\n")
        
        results = []
        for question in test_questions:
            result = test_question(rag, question, show_sources=False)
            results.append({
                'question': question,
                'answer': result['answer'],
                'time': result['generation_time']
            })
            print("\n" + "-"*80)
        
        # Summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        avg_time = sum(r['time'] for r in results) / len(results)
        print(f"\nTotal questions tested: {len(results)}")
        print(f"Average generation time: {avg_time:.2f}s")
        
        print("\n📝 All Answers:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Q: {r['question']}")
            print(f"   A: {r['answer'][:200]}...")
        
        # Save results
        with open('backend_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✅ Results saved to: backend_test_results.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
