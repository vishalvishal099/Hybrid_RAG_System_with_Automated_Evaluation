"""
Question Generation for Evaluation
Generate 100 diverse Q&A pairs from Wikipedia corpus
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import yaml
from tqdm import tqdm
import re

# For question generation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class QuestionGenerator:
    """Generate evaluation questions from corpus"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.corpus_chunks = []
        self.documents = []
        self.qa_pairs = []
    
    def load_corpus(self, corpus_path: str = None):
        """Load the processed corpus"""
        if corpus_path is None:
            corpus_path = self.config['data']['corpus_file']
        
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        
        self.corpus_chunks = corpus['chunks']
        self.documents = corpus['documents']
        
        print(f"✓ Loaded {len(self.corpus_chunks)} chunks from {len(self.documents)} documents")
    
    def generate_factual_questions(self, num_questions: int = 30) -> List[Dict]:
        """
        Generate factual questions (who, what, when, where)
        Using rule-based extraction
        """
        questions = []
        patterns = {
            'what_is': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+([^.]+)',
            'who': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:founded|created|discovered|invented|wrote)\s+([^.]+)',
            'when': r'In\s+(\d{4}),\s+([^.]+)',
            'where': r'(?:located|situated|found)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        }
        
        print("Generating factual questions...")
        
        # Sample diverse chunks
        sampled_chunks = random.sample(
            self.corpus_chunks,
            min(len(self.corpus_chunks), num_questions * 3)
        )
        
        for chunk in sampled_chunks:
            if len(questions) >= num_questions:
                break
            
            text = chunk['text']
            sentences = text.split('.')
            
            for sentence in sentences:
                if len(questions) >= num_questions:
                    break
                
                # Try different patterns
                if 'is' in sentence.lower() or 'are' in sentence.lower():
                    # Extract subject
                    words = sentence.split()
                    if len(words) > 5:
                        # Simple what is question
                        subject = ' '.join(words[:3])
                        if subject[0].isupper():
                            question = f"What is {subject}?"
                            questions.append({
                                'question_id': f"factual_{len(questions)}",
                                'question': question,
                                'ground_truth': sentence.strip(),
                                'source_url': chunk['url'],
                                'source_title': chunk['title'],
                                'chunk_id': chunk['chunk_id'],
                                'question_type': 'factual',
                                'difficulty': 'easy'
                            })
        
        return questions
    
    def generate_questions_with_llm(self, num_questions: int = 70) -> List[Dict]:
        """
        Generate diverse questions using T5 or similar model
        """
        questions = []
        
        print("Loading question generation model...")
        # Use a smaller model for question generation
        model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                model = model.cuda()
        except:
            print("Warning: Could not load question generation model, using rule-based fallback")
            return self.generate_rule_based_questions(num_questions)
        
        print(f"Generating {num_questions} questions with LLM...")
        
        # Sample diverse documents
        sampled_docs = random.sample(
            self.documents,
            min(len(self.documents), num_questions)
        )
        
        for doc in tqdm(sampled_docs[:num_questions]):
            # Find chunks for this document
            doc_chunks = [c for c in self.corpus_chunks if c['url'] == doc['url']]
            
            if not doc_chunks:
                continue
            
            # Use first chunk as context
            chunk = doc_chunks[0]
            context = chunk['text'][:500]  # Limit context
            
            # Generate question
            input_text = f"generate question: {context}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
            
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Determine question type
            q_lower = question.lower()
            if any(word in q_lower for word in ['compare', 'difference', 'similar']):
                q_type = 'comparative'
                difficulty = 'medium'
            elif any(word in q_lower for word in ['why', 'how', 'explain']):
                q_type = 'inferential'
                difficulty = 'hard'
            else:
                q_type = 'factual'
                difficulty = 'easy'
            
            questions.append({
                'question_id': f"llm_{len(questions)}",
                'question': question,
                'ground_truth': context[:200],  # First part of context as ground truth
                'source_url': chunk['url'],
                'source_title': chunk['title'],
                'chunk_id': chunk['chunk_id'],
                'question_type': q_type,
                'difficulty': difficulty
            })
        
        return questions
    
    def generate_rule_based_questions(self, num_questions: int) -> List[Dict]:
        """
        Fallback: Generate questions using templates and corpus
        """
        questions = []
        
        templates = {
            'what': [
                "What is {entity}?",
                "What are the main features of {entity}?",
                "What does {entity} mean?",
            ],
            'who': [
                "Who created {entity}?",
                "Who discovered {entity}?",
                "Who is associated with {entity}?",
            ],
            'when': [
                "When was {entity} established?",
                "When did {entity} happen?",
            ],
            'where': [
                "Where is {entity} located?",
                "Where did {entity} originate?",
            ],
            'how': [
                "How does {entity} work?",
                "How is {entity} used?",
            ],
            'why': [
                "Why is {entity} important?",
                "Why was {entity} created?",
            ]
        }
        
        print(f"Generating {num_questions} rule-based questions...")
        
        # Sample documents
        sampled_docs = random.sample(
            self.documents,
            min(len(self.documents), num_questions)
        )
        
        for doc in sampled_docs[:num_questions]:
            # Get entity (title)
            entity = doc['title']
            
            # Choose random template type
            template_type = random.choice(list(templates.keys()))
            template = random.choice(templates[template_type])
            
            question = template.format(entity=entity)
            
            # Find a chunk for ground truth
            doc_chunks = [c for c in self.corpus_chunks if c['url'] == doc['url']]
            if doc_chunks:
                chunk = doc_chunks[0]
                ground_truth = chunk['text'][:200]
                
                # Determine question type
                if template_type in ['what', 'who', 'when', 'where']:
                    q_type = 'factual'
                    difficulty = 'easy'
                elif template_type == 'how':
                    q_type = 'inferential'
                    difficulty = 'medium'
                else:  # why
                    q_type = 'inferential'
                    difficulty = 'hard'
                
                questions.append({
                    'question_id': f"rule_{len(questions)}",
                    'question': question,
                    'ground_truth': ground_truth,
                    'source_url': chunk['url'],
                    'source_title': chunk['title'],
                    'chunk_id': chunk['chunk_id'],
                    'question_type': q_type,
                    'difficulty': difficulty
                })
        
        return questions
    
    def generate_multi_hop_questions(self, num_questions: int = 10) -> List[Dict]:
        """
        Generate multi-hop questions requiring multiple documents
        """
        questions = []
        
        print(f"Generating {num_questions} multi-hop questions...")
        
        # Sample pairs of related documents
        for i in range(min(num_questions, len(self.documents) // 2)):
            doc1 = random.choice(self.documents)
            doc2 = random.choice(self.documents)
            
            if doc1['url'] != doc2['url']:
                question = f"How are {doc1['title']} and {doc2['title']} related?"
                
                # Get chunks
                chunks1 = [c for c in self.corpus_chunks if c['url'] == doc1['url']]
                chunks2 = [c for c in self.corpus_chunks if c['url'] == doc2['url']]
                
                if chunks1 and chunks2:
                    ground_truth = f"Information from {doc1['title']} and {doc2['title']}"
                    
                    questions.append({
                        'question_id': f"multi_hop_{len(questions)}",
                        'question': question,
                        'ground_truth': ground_truth,
                        'source_url': [doc1['url'], doc2['url']],
                        'source_title': [doc1['title'], doc2['title']],
                        'chunk_id': [chunks1[0]['chunk_id'], chunks2[0]['chunk_id']],
                        'question_type': 'multi_hop',
                        'difficulty': 'hard'
                    })
        
        return questions
    
    def generate_comparative_questions(self, num_questions: int = 10) -> List[Dict]:
        """
        Generate comparative questions
        """
        questions = []
        
        print(f"Generating {num_questions} comparative questions...")
        
        templates = [
            "What is the difference between {entity1} and {entity2}?",
            "Compare {entity1} and {entity2}.",
            "How do {entity1} and {entity2} differ?",
            "What are the similarities between {entity1} and {entity2}?",
        ]
        
        for i in range(min(num_questions, len(self.documents) // 2)):
            doc1 = random.choice(self.documents)
            doc2 = random.choice(self.documents)
            
            if doc1['url'] != doc2['url']:
                template = random.choice(templates)
                question = template.format(
                    entity1=doc1['title'],
                    entity2=doc2['title']
                )
                
                chunks1 = [c for c in self.corpus_chunks if c['url'] == doc1['url']]
                chunks2 = [c for c in self.corpus_chunks if c['url'] == doc2['url']]
                
                if chunks1 and chunks2:
                    ground_truth = f"Comparison of {doc1['title']} and {doc2['title']}"
                    
                    questions.append({
                        'question_id': f"comparative_{len(questions)}",
                        'question': question,
                        'ground_truth': ground_truth,
                        'source_url': [doc1['url'], doc2['url']],
                        'source_title': [doc1['title'], doc2['title']],
                        'chunk_id': [chunks1[0]['chunk_id'], chunks2[0]['chunk_id']],
                        'question_type': 'comparative',
                        'difficulty': 'medium'
                    })
        
        return questions
    
    def generate_all_questions(self, num_total: int = 100) -> List[Dict]:
        """
        Generate diverse set of questions
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION QUESTIONS")
        print("="*60)
        
        all_questions = []
        
        # Distribution of question types
        num_factual = 40
        num_inferential = 30
        num_comparative = 15
        num_multi_hop = 15
        
        # Generate factual questions
        factual = self.generate_factual_questions(num_factual)
        all_questions.extend(factual[:num_factual])
        
        # Generate inferential questions (using LLM or rule-based)
        inferential = self.generate_rule_based_questions(num_inferential)
        all_questions.extend(inferential[:num_inferential])
        
        # Generate comparative questions
        comparative = self.generate_comparative_questions(num_comparative)
        all_questions.extend(comparative[:num_comparative])
        
        # Generate multi-hop questions
        multi_hop = self.generate_multi_hop_questions(num_multi_hop)
        all_questions.extend(multi_hop[:num_multi_hop])
        
        # Shuffle
        random.shuffle(all_questions)
        
        # Reassign IDs
        for i, q in enumerate(all_questions):
            q['question_id'] = f"Q{i+1:03d}"
        
        print(f"\n✓ Generated {len(all_questions)} questions")
        print(f"  Factual: {len([q for q in all_questions if q['question_type'] == 'factual'])}")
        print(f"  Inferential: {len([q for q in all_questions if q['question_type'] == 'inferential'])}")
        print(f"  Comparative: {len([q for q in all_questions if q['question_type'] == 'comparative'])}")
        print(f"  Multi-hop: {len([q for q in all_questions if q['question_type'] == 'multi_hop'])}")
        
        return all_questions[:num_total]
    
    def save_questions(self, questions: List[Dict], output_path: str = None):
        """Save questions to JSON file"""
        if output_path is None:
            output_path = self.config['paths']['questions_dataset']
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'total_questions': len(questions),
                'question_types': {
                    q_type: len([q for q in questions if q['question_type'] == q_type])
                    for q_type in set(q['question_type'] for q in questions)
                }
            },
            'questions': questions
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Saved questions to {output_path}")


def main():
    """Main execution"""
    generator = QuestionGenerator()
    
    # Load corpus
    generator.load_corpus()
    
    # Generate questions
    questions = generator.generate_all_questions(num_total=100)
    
    # Save questions
    generator.save_questions(questions)
    
    print("\n" + "="*60)
    print("QUESTION GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
