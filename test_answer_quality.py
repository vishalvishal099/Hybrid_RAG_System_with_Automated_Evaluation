"""
Test script to evaluate current answer quality
"""
from src.rag_system import HybridRAGSystem
import json

# Initialize RAG system
print('Loading RAG system...')
rag = HybridRAGSystem()
rag.load_corpus()
rag.load_indexes()
print('✓ System loaded.\n')

# Test with diverse question types
test_questions = [
    {
        'question': 'What is artificial intelligence?',
        'type': 'factual'
    },
    {
        'question': 'Who invented the telephone?',
        'type': 'factual'
    },
    {
        'question': 'What is Specially protected areas?',
        'type': 'factual'
    },
    {
        'question': 'How does photosynthesis work?',
        'type': 'process'
    }
]

results_summary = []

for item in test_questions:
    q = item['question']
    print(f"\n{'='*80}")
    print(f"Q: {q}")
    print(f"Type: {item['type']}")
    print('='*80)
    
    # Retrieve
    retrieval_results = rag.retrieve(q, method='hybrid')
    chunks = retrieval_results['chunks']
    print(f'\n✓ Retrieved {len(chunks)} chunks')
    
    # Show top source
    if chunks:
        print(f"  Top source: {chunks[0].get('title', 'N/A')} (Score: {chunks[0].get('rrf_score', 0):.4f})")
    
    # Generate answer
    answer_data = rag.generate_answer(q, chunks)
    answer = answer_data['answer']
    gen_time = answer_data['generation_time']
    
    print(f'\n📝 ANSWER ({gen_time:.2f}s, {len(answer)} chars):')
    print('-' * 80)
    print(answer)
    print('-' * 80)
    
    # Quality checks
    quality_issues = []
    if len(answer) < 30:
        quality_issues.append('❌ Too short')
    if len(answer) > 500:
        quality_issues.append('⚠️ Too long')
    if answer.count('.') < 1:
        quality_issues.append('❌ Incomplete (no sentence endings)')
    if 'cannot answer' in answer.lower():
        quality_issues.append('❌ Failed to answer')
    if answer.count('[') > 2:
        quality_issues.append('❌ Too many references')
    
    if not quality_issues:
        quality_issues.append('✅ Quality OK')
    
    print('\nQuality Assessment:')
    for issue in quality_issues:
        print(f'  {issue}')
    
    results_summary.append({
        'question': q,
        'answer_length': len(answer),
        'generation_time': gen_time,
        'quality_issues': quality_issues
    })

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY")
print('='*80)
avg_length = sum(r['answer_length'] for r in results_summary) / len(results_summary)
avg_time = sum(r['generation_time'] for r in results_summary) / len(results_summary)

print(f"\nAverage answer length: {avg_length:.0f} chars")
print(f"Average generation time: {avg_time:.2f}s")

print("\nPer-question results:")
for r in results_summary:
    print(f"\n  Q: {r['question'][:50]}...")
    print(f"     Length: {r['answer_length']} | Time: {r['generation_time']:.2f}s")
    print(f"     Issues: {', '.join(r['quality_issues'])}")
