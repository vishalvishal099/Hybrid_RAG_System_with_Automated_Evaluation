"""
Generate Adversarial Questions for Testing RAG Robustness
Creates ambiguous, negated, paraphrased, and unanswerable questions
"""

import json
import random

def load_original_questions():
    """Load original questions"""
    with open('data/questions_100.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']

def generate_ambiguous_questions(questions):
    """Generate ambiguous questions by removing key context"""
    ambiguous = []
    samples = random.sample(questions, min(10, len(questions)))
    
    for q in samples:
        original = q['question']
        # Remove specific entity names to make ambiguous
        ambiguous_version = original.replace('What is', 'What are they referring to when discussing')
        ambiguous_version = ambiguous_version.replace('Who', 'Which entity')
        
        ambiguous.append({
            'question': ambiguous_version,
            'source_url': q['source_url'],
            'ground_truth': q.get('ground_truth', ''),
            'type': 'ambiguous',
            'original_question': original
        })
    
    return ambiguous

def generate_negated_questions(questions):
    """Generate negated questions"""
    negated = []
    samples = random.sample(questions, min(10, len(questions)))
    
    for q in samples:
        original = q['question']
        # Add negation
        if 'What is' in original:
            negated_version = original.replace('What is', 'What is NOT')
        elif 'Who' in original:
            negated_version = original.replace('Who', 'Who is NOT')
        elif 'How' in original:
            negated_version = original.replace('How', 'How NOT')
        else:
            negated_version = f"What is NOT true about {original[10:]}"
        
        negated.append({
            'question': negated_version,
            'source_url': q['source_url'],
            'ground_truth': 'NEGATED: ' + q.get('ground_truth', ''),
            'type': 'negated',
            'original_question': original
        })
    
    return negated

def generate_paraphrased_questions(questions):
    """Generate paraphrased questions"""
    paraphrased = []
    samples = random.sample(questions, min(10, len(questions)))
    
    paraphrase_templates = [
        lambda q: q.replace('What is', 'Can you explain what'),
        lambda q: q.replace('Who', 'Which person or entity'),
        lambda q: q.replace('How', 'In what way'),
        lambda q: f"Could you tell me about {q[10:]}",
        lambda q: f"I'm curious to know {q.lower()}",
    ]
    
    for q in samples:
        original = q['question']
        template = random.choice(paraphrase_templates)
        try:
            paraphrased_version = template(original)
        except:
            paraphrased_version = f"Please explain {original[10:]}"
        
        paraphrased.append({
            'question': paraphrased_version,
            'source_url': q['source_url'],
            'ground_truth': q.get('ground_truth', ''),
            'type': 'paraphrased',
            'original_question': original
        })
    
    return paraphrased

def generate_unanswerable_questions():
    """Generate unanswerable questions (not in corpus)"""
    unanswerable = [
        {
            'question': 'What is the current Bitcoin price in real-time?',
            'source_url': 'NONE',
            'ground_truth': 'This information is not available in the corpus.',
            'type': 'unanswerable',
            'reason': 'Real-time data not in static corpus'
        },
        {
            'question': 'What will happen in the year 2050?',
            'source_url': 'NONE',
            'ground_truth': 'Future predictions are not available.',
            'type': 'unanswerable',
            'reason': 'Future prediction'
        },
        {
            'question': 'What is my personal opinion on this topic?',
            'source_url': 'NONE',
            'ground_truth': 'Personal opinions are subjective.',
            'type': 'unanswerable',
            'reason': 'Subjective/personal question'
        },
        {
            'question': 'What is the secret password for accessing this system?',
            'source_url': 'NONE',
            'ground_truth': 'Security information is not available.',
            'type': 'unanswerable',
            'reason': 'Security-related'
        },
        {
            'question': 'How do I hack into a computer system?',
            'source_url': 'NONE',
            'ground_truth': 'Such information is not provided.',
            'type': 'unanswerable',
            'reason': 'Harmful/illegal content'
        },
        {
            'question': 'What is the meaning of the fictional concept "Flibbertigibbet-3000"?',
            'source_url': 'NONE',
            'ground_truth': 'This fictional concept does not exist in the corpus.',
            'type': 'unanswerable',
            'reason': 'Non-existent entity'
        },
        {
            'question': 'Can you perform a mathematical calculation: 123456 * 789012?',
            'source_url': 'NONE',
            'ground_truth': 'The corpus does not contain computational results.',
            'type': 'unanswerable',
            'reason': 'Computation request'
        },
        {
            'question': 'What does this image show: [image.jpg]?',
            'source_url': 'NONE',
            'ground_truth': 'Image analysis is not supported.',
            'type': 'unanswerable',
            'reason': 'Multimodal input'
        },
        {
            'question': 'What is happening right now in New York City?',
            'source_url': 'NONE',
            'ground_truth': 'Current events are not in the static corpus.',
            'type': 'unanswerable',
            'reason': 'Current events'
        },
        {
            'question': 'Tell me everything about XYZ-9999, a topic that does not exist.',
            'source_url': 'NONE',
            'ground_truth': 'This topic does not exist in the knowledge base.',
            'type': 'unanswerable',
            'reason': 'Non-existent topic'
        }
    ]
    
    return unanswerable

def main():
    """Generate all adversarial questions"""
    print("🔄 Generating adversarial questions...")
    
    # Load original questions
    questions = load_original_questions()
    print(f"✓ Loaded {len(questions)} original questions")
    
    # Generate adversarial questions
    ambiguous = generate_ambiguous_questions(questions)
    print(f"✓ Generated {len(ambiguous)} ambiguous questions")
    
    negated = generate_negated_questions(questions)
    print(f"✓ Generated {len(negated)} negated questions")
    
    paraphrased = generate_paraphrased_questions(questions)
    print(f"✓ Generated {len(paraphrased)} paraphrased questions")
    
    unanswerable = generate_unanswerable_questions()
    print(f"✓ Generated {len(unanswerable)} unanswerable questions")
    
    # Combine all adversarial questions
    all_adversarial = ambiguous + negated + paraphrased + unanswerable
    
    # Save to file
    output = {
        'description': 'Adversarial questions for testing RAG robustness',
        'total_questions': len(all_adversarial),
        'breakdown': {
            'ambiguous': len(ambiguous),
            'negated': len(negated),
            'paraphrased': len(paraphrased),
            'unanswerable': len(unanswerable)
        },
        'questions': all_adversarial
    }
    
    with open('data/adversarial_questions.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(all_adversarial)} adversarial questions to data/adversarial_questions.json")
    print(f"\nBreakdown:")
    print(f"  - Ambiguous: {len(ambiguous)}")
    print(f"  - Negated: {len(negated)}")
    print(f"  - Paraphrased: {len(paraphrased)}")
    print(f"  - Unanswerable: {len(unanswerable)}")

if __name__ == "__main__":
    main()
