"""
Novel Metrics for RAG System Evaluation
Implements 4 advanced metrics:
1. Entity Coverage - % of entities from context in answer
2. Answer Diversity - Lexical diversity measure  
3. Hallucination Rate - % of facts not supported by context
4. Temporal Consistency - Correct time references
"""

import re
import json
from typing import List, Dict, Set
from datetime import datetime
from collections import Counter


class NovelMetrics:
    """
    Advanced evaluation metrics for RAG systems
    """
    
    def __init__(self):
        # Common time indicators
        self.time_patterns = [
            r'\b(19|20)\d{2}\b',  # Years 1900-2099
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
            r'\b\d{1,2}\s+(days?|weeks?|months?|years?)\s+(ago|later|before|after)\b',
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(century|decade|millennium)\b',
        ]
    
    def extract_entities_regex(self, text: str) -> Set[str]:
        """
        Extract entities using regex patterns (without spaCy)
        Captures: Proper nouns, acronyms, numbers
        """
        entities = set()
        
        # Pattern 1: Capitalized words (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(proper_nouns)
        
        # Pattern 2: All-caps words (acronyms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.update(acronyms)
        
        # Pattern 3: Numbers with units
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:km|m|kg|g|miles|pounds|dollars|%|percent)?\b', text)
        entities.update(numbers)
        
        # Filter out common words
        stopwords = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'And', 'Or', 'But'}
        entities = {e for e in entities if e not in stopwords and len(e) > 1}
        
        return entities
    
    def calculate_entity_coverage(
        self,
        answer: str,
        context_chunks: List[str]
    ) -> Dict[str, float]:
        """
        Calculate what % of entities from context appear in the answer
        
        High coverage = answer uses information from context
        Low coverage = answer may be generic or hallucinated
        
        Returns:
            {
                'entity_coverage': 0.0-1.0,
                'context_entities_count': int,
                'answer_entities_count': int,
                'shared_entities_count': int
            }
        """
        if not answer or not context_chunks:
            return {
                'entity_coverage': 0.0,
                'context_entities_count': 0,
                'answer_entities_count': 0,
                'shared_entities_count': 0
            }
        
        # Extract entities
        context_text = ' '.join(context_chunks)
        context_entities = self.extract_entities_regex(context_text)
        answer_entities = self.extract_entities_regex(answer)
        
        # Find overlap
        shared_entities = context_entities.intersection(answer_entities)
        
        # Coverage = shared / context (how much of context is used)
        if context_entities:
            coverage = len(shared_entities) / len(context_entities)
        else:
            coverage = 0.0
        
        return {
            'entity_coverage': round(coverage, 4),
            'context_entities_count': len(context_entities),
            'answer_entities_count': len(answer_entities),
            'shared_entities_count': len(shared_entities),
            'context_entities': list(context_entities)[:10],  # Sample
            'answer_entities': list(answer_entities)[:10]  # Sample
        }
    
    def calculate_answer_diversity(self, answer: str) -> Dict[str, float]:
        """
        Calculate lexical diversity of answer
        
        Metric: Type-Token Ratio (TTR) and related measures
        - TTR = unique words / total words
        - High diversity = varied vocabulary
        - Low diversity = repetitive
        
        Returns:
            {
                'ttr': Type-Token Ratio,
                'unique_words': count,
                'total_words': count,
                'average_word_length': float
            }
        """
        if not answer:
            return {
                'ttr': 0.0,
                'unique_words': 0,
                'total_words': 0,
                'average_word_length': 0.0
            }
        
        # Tokenize
        words = re.findall(r'\b\w+\b', answer.lower())
        
        if not words:
            return {
                'ttr': 0.0,
                'unique_words': 0,
                'total_words': 0,
                'average_word_length': 0.0
            }
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        avg_word_len = sum(len(w) for w in words) / len(words)
        
        return {
            'ttr': round(ttr, 4),
            'unique_words': len(unique_words),
            'total_words': len(words),
            'average_word_length': round(avg_word_len, 2)
        }
    
    def calculate_hallucination_rate(
        self,
        answer: str,
        context_chunks: List[str]
    ) -> Dict[str, float]:
        """
        Estimate hallucination rate (claims not supported by context)
        
        Method: N-gram matching
        - Extract important phrases (3-5 word n-grams) from answer
        - Check how many appear (or have high overlap) with context
        - Unsupported phrases = potential hallucinations
        
        Returns:
            {
                'hallucination_rate': 0.0-1.0 (lower is better),
                'total_claims': int,
                'unsupported_claims': int,
                'supported_claims': int
            }
        """
        if not answer or not context_chunks:
            return {
                'hallucination_rate': 0.0,
                'total_claims': 0,
                'unsupported_claims': 0,
                'supported_claims': 0
            }
        
        context_text = ' '.join(context_chunks).lower()
        answer_lower = answer.lower()
        
        # Extract claims as trigrams (3-word phrases)
        answer_words = re.findall(r'\b\w+\b', answer_lower)
        
        if len(answer_words) < 3:
            return {
                'hallucination_rate': 0.0,
                'total_claims': 0,
                'unsupported_claims': 0,
                'supported_claims': 0
            }
        
        # Generate trigrams
        trigrams = []
        for i in range(len(answer_words) - 2):
            trigram = ' '.join(answer_words[i:i+3])
            trigrams.append(trigram)
        
        # Check support in context
        supported = 0
        unsupported = 0
        
        for trigram in trigrams:
            # Check if trigram or 2 of its 3 words appear in context
            words_in_trigram = trigram.split()
            matches = sum(1 for w in words_in_trigram if w in context_text)
            
            if matches >= 2:  # At least 2/3 words found
                supported += 1
            else:
                unsupported += 1
        
        total_claims = len(trigrams)
        hallucination_rate = unsupported / total_claims if total_claims > 0 else 0.0
        
        return {
            'hallucination_rate': round(hallucination_rate, 4),
            'total_claims': total_claims,
            'unsupported_claims': unsupported,
            'supported_claims': supported
        }
    
    def calculate_temporal_consistency(
        self,
        answer: str,
        context_chunks: List[str] = None
    ) -> Dict[str, any]:
        """
        Check temporal consistency of answer
        
        Checks:
        1. Presence of time references
        2. Format consistency (e.g., all years in same format)
        3. Logical ordering (if comparing times)
        
        Returns:
            {
                'has_temporal_info': bool,
                'temporal_references_count': int,
                'years_found': list,
                'months_found': list,
                'is_consistent': bool
            }
        """
        if not answer:
            return {
                'has_temporal_info': False,
                'temporal_references_count': 0,
                'years_found': [],
                'months_found': [],
                'is_consistent': True
            }
        
        # Find all temporal references
        years = re.findall(r'\b(19|20)\d{2}\b', answer)
        months = re.findall(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
            answer
        )
        
        temporal_count = len(years) + len(months)
        has_temporal = temporal_count > 0
        
        # Check consistency: years should be in ascending order if multiple
        is_consistent = True
        if len(years) > 1:
            years_int = [int(y) for y in years]
            # Check if sorted or reverse sorted
            is_consistent = (years_int == sorted(years_int) or 
                           years_int == sorted(years_int, reverse=True))
        
        return {
            'has_temporal_info': has_temporal,
            'temporal_references_count': temporal_count,
            'years_found': years,
            'months_found': months,
            'is_consistent': is_consistent
        }
    
    def evaluate_all(
        self,
        answer: str,
        context_chunks: List[str]
    ) -> Dict:
        """
        Run all 4 novel metrics
        
        Returns comprehensive metrics dictionary
        """
        return {
            'entity_coverage': self.calculate_entity_coverage(answer, context_chunks),
            'answer_diversity': self.calculate_answer_diversity(answer),
            'hallucination': self.calculate_hallucination_rate(answer, context_chunks),
            'temporal_consistency': self.calculate_temporal_consistency(answer, context_chunks)
        }


def main():
    """Test novel metrics"""
    print("="*70)
    print("TESTING NOVEL METRICS")
    print("="*70)
    
    metrics = NovelMetrics()
    
    # Test case
    context = [
        "Albert Einstein was born in 1879 in Germany. He developed the theory of relativity in 1905.",
        "Einstein won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect."
    ]
    
    answer = "Einstein, born in 1879, developed relativity in 1905 and won the Nobel Prize in 1921."
    
    print(f"\nContext: {' '.join(context)[:150]}...")
    print(f"Answer: {answer}\n")
    
    results = metrics.evaluate_all(answer, context)
    
    print("📊 Novel Metrics Results:\n")
    
    print("1. Entity Coverage:")
    ec = results['entity_coverage']
    print(f"   Coverage: {ec['entity_coverage']:.2%}")
    print(f"   Context entities: {ec['context_entities_count']}")
    print(f"   Answer entities: {ec['answer_entities_count']}")
    print(f"   Shared: {ec['shared_entities_count']}")
    
    print("\n2. Answer Diversity:")
    ad = results['answer_diversity']
    print(f"   TTR: {ad['ttr']:.3f}")
    print(f"   Unique words: {ad['unique_words']}/{ad['total_words']}")
    print(f"   Avg word length: {ad['average_word_length']:.1f} chars")
    
    print("\n3. Hallucination Rate:")
    hr = results['hallucination']
    print(f"   Hallucination rate: {hr['hallucination_rate']:.2%}")
    print(f"   Total claims: {hr['total_claims']}")
    print(f"   Unsupported: {hr['unsupported_claims']}")
    print(f"   Supported: {hr['supported_claims']}")
    
    print("\n4. Temporal Consistency:")
    tc = results['temporal_consistency']
    print(f"   Has temporal info: {tc['has_temporal_info']}")
    print(f"   References found: {tc['temporal_references_count']}")
    print(f"   Years: {tc['years_found']}")
    print(f"   Is consistent: {tc['is_consistent']}")
    
    print("\n" + "="*70)
    print("✅ Novel metrics test complete!")


if __name__ == "__main__":
    main()
