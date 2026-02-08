"""
Evaluation Metrics for Hybrid RAG System
Implements MRR (mandatory) + BERTScore + NDCG@K (custom metrics)
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import yaml
from collections import defaultdict

# Evaluation metrics
from sklearn.metrics import ndcg_score
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import re


class RAGEvaluator:
    """
    Comprehensive evaluation for RAG system
    
    Metrics:
    1. MRR (Mandatory) - URL-level Mean Reciprocal Rank
    2. BERTScore (Custom 1) - Semantic similarity of answers
    3. NDCG@K (Custom 2) - Ranking quality of retrieved documents
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_mrr(self, retrieved_urls: List[str], ground_truth_urls: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) at URL level
        
        Metric Justification:
        MRR measures how quickly the system identifies the correct source document.
        It's crucial for RAG systems because finding the right source document early
        in the ranking directly impacts answer quality. A high MRR means users get
        relevant information faster.
        
        Calculation:
        For each query, find the rank position of the first correct URL in retrieved results.
        MRR = 1/rank if found, 0 if not found
        
        Interpretation:
        - MRR = 1.0: Perfect, correct URL always ranked first
        - MRR = 0.5: Correct URL appears at rank 2 on average
        - MRR = 0.0: No correct URLs retrieved
        - MRR > 0.7: Excellent retrieval performance
        - MRR 0.5-0.7: Good performance
        - MRR < 0.5: Needs improvement
        
        Args:
            retrieved_urls: List of URLs in order retrieved
            ground_truth_urls: List or single URL that are correct
        
        Returns:
            Reciprocal rank (0 if not found)
        """
        # Normalize ground truth to list
        if isinstance(ground_truth_urls, str):
            ground_truth_urls = [ground_truth_urls]
        
        # Find first matching URL
        for rank, url in enumerate(retrieved_urls, start=1):
            if url in ground_truth_urls:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore for semantic similarity
        
        Metric Justification:
        BERTScore measures semantic similarity between generated answers and ground truth
        using contextual embeddings. Unlike lexical metrics (ROUGE, BLEU), it captures
        meaning even when wording differs. This is critical for RAG systems where the
        same information can be expressed in multiple ways.
        
        Calculation:
        1. Compute BERT embeddings for tokens in prediction and reference
        2. Calculate cosine similarity between all token pairs
        3. Use greedy matching to find optimal alignment
        4. Compute Precision, Recall, and F1 scores
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Interpretation:
        - F1 > 0.9: Excellent semantic match, answer conveys same meaning
        - F1 0.8-0.9: Good match, minor semantic differences
        - F1 0.7-0.8: Moderate match, some information missing or incorrect
        - F1 < 0.7: Poor match, significant semantic divergence
        
        Precision measures how much of the generated answer is relevant
        Recall measures how much of the ground truth is covered
        
        Args:
            predictions: Generated answers
            references: Ground truth answers
        
        Returns:
            Dict with precision, recall, and F1 scores
        """
        if not predictions or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate BERTScore
        P, R, F1 = bert_score(
            predictions,
            references,
            lang='en',
            rescale_with_baseline=True,
            verbose=False
        )
        
        return {
            'precision': float(P.mean()),
            'recall': float(R.mean()),
            'f1': float(F1.mean())
        }
    
    def calculate_ndcg(self, retrieved_urls: List[str], ground_truth_urls: List[str], k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Metric Justification:
        NDCG@K evaluates ranking quality by considering both relevance and position.
        It's essential for RAG because:
        1. Position matters - earlier results are more valuable
        2. Graded relevance - not binary (relevant vs irrelevant)
        3. Penalizes putting relevant docs lower in ranking
        
        For RAG, good ranking means relevant chunks appear early, providing
        better context to the LLM and improving answer quality.
        
        Calculation:
        DCG@K = Σ(i=1 to k) [rel_i / log2(i+1)]
        IDCG@K = DCG@K of ideal ranking (all relevant docs first)
        NDCG@K = DCG@K / IDCG@K
        
        where rel_i = 1 if URL at position i is relevant, 0 otherwise
        
        Interpretation:
        - NDCG = 1.0: Perfect ranking, all relevant URLs at top
        - NDCG = 0.8-1.0: Excellent ranking quality
        - NDCG = 0.6-0.8: Good ranking, some relevant docs further down
        - NDCG = 0.4-0.6: Moderate ranking, relevant docs scattered
        - NDCG < 0.4: Poor ranking, relevant docs buried or missing
        
        Higher NDCG indicates better ranking, meaning the LLM receives
        more relevant context earlier, improving generation quality.
        
        Args:
            retrieved_urls: List of URLs in retrieval order
            ground_truth_urls: Relevant URLs
            k: Cutoff position (default 5)
        
        Returns:
            NDCG@K score
        """
        # Normalize ground truth to list
        if isinstance(ground_truth_urls, str):
            ground_truth_urls = [ground_truth_urls]
        
        # Create relevance scores (1 if relevant, 0 otherwise)
        relevance_scores = []
        for url in retrieved_urls[:k]:
            if url in ground_truth_urls:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)
        
        # Pad to k if needed
        while len(relevance_scores) < k:
            relevance_scores.append(0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Calculate ideal DCG (all relevant docs at top)
        num_relevant = min(len(ground_truth_urls), k)
        idcg = 0.0
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        # Return NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_precision_recall(self, retrieved_urls: List[str], 
                                   ground_truth_urls: List[str], 
                                   k: int = 5) -> Tuple[float, float]:
        """
        Calculate Precision@K and Recall@K
        
        Args:
            retrieved_urls: Retrieved URLs
            ground_truth_urls: Relevant URLs
            k: Cutoff
        
        Returns:
            (precision@k, recall@k)
        """
        if isinstance(ground_truth_urls, str):
            ground_truth_urls = [ground_truth_urls]
        
        retrieved_k = set(retrieved_urls[:k])
        relevant = set(ground_truth_urls)
        
        if not retrieved_k:
            return 0.0, 0.0
        
        true_positives = len(retrieved_k.intersection(relevant))
        
        precision = true_positives / k if k > 0 else 0.0
        recall = true_positives / len(relevant) if relevant else 0.0
        
        return precision, recall
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate_single_query(self, question: Dict, rag_response: Dict) -> Dict:
        """
        Evaluate a single query
        
        Args:
            question: Question dict with ground_truth and source_url
            rag_response: RAG system response with answer and sources
        
        Returns:
            Dict with all evaluation metrics
        """
        # Extract URLs from response
        retrieved_urls = [source['url'] for source in rag_response['sources']]
        
        # Ground truth URLs
        gt_urls = question.get('source_url')
        if isinstance(gt_urls, str):
            gt_urls = [gt_urls]
        
        # 1. MRR (Mandatory)
        mrr = self.calculate_mrr(retrieved_urls, gt_urls)
        
        # 2. NDCG@5 (Custom Metric 1)
        ndcg = self.calculate_ndcg(retrieved_urls, gt_urls, k=5)
        
        # 3. Precision and Recall
        precision, recall = self.calculate_precision_recall(retrieved_urls, gt_urls, k=5)
        
        # 4. BERTScore (Custom Metric 2)
        bert_scores = self.calculate_bertscore(
            [rag_response['answer']],
            [question['ground_truth']]
        )
        
        # 5. ROUGE scores
        rouge_scores = self.calculate_rouge(
            rag_response['answer'],
            question['ground_truth']
        )
        
        return {
            'question_id': question['question_id'],
            'question': question['question'],
            'ground_truth': question['ground_truth'],
            'generated_answer': rag_response['answer'],
            'question_type': question.get('question_type', 'unknown'),
            'difficulty': question.get('difficulty', 'unknown'),
            'metrics': {
                'mrr': mrr,
                'ndcg_5': ndcg,
                'precision_5': precision,
                'recall_5': recall,
                'bertscore_f1': bert_scores['f1'],
                'bertscore_precision': bert_scores['precision'],
                'bertscore_recall': bert_scores['recall'],
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL']
            },
            'retrieval_time': rag_response['metadata']['retrieval_time'],
            'generation_time': rag_response['metadata']['generation_time'],
            'total_time': rag_response['metadata']['total_time'],
            'retrieved_urls': retrieved_urls[:5],
            'ground_truth_urls': gt_urls
        }
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate evaluation results across all questions
        
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        # Overall metrics
        metrics = {
            'mrr': np.mean([r['metrics']['mrr'] for r in results]),
            'ndcg_5': np.mean([r['metrics']['ndcg_5'] for r in results]),
            'precision_5': np.mean([r['metrics']['precision_5'] for r in results]),
            'recall_5': np.mean([r['metrics']['recall_5'] for r in results]),
            'bertscore_f1': np.mean([r['metrics']['bertscore_f1'] for r in results]),
            'bertscore_precision': np.mean([r['metrics']['bertscore_precision'] for r in results]),
            'bertscore_recall': np.mean([r['metrics']['bertscore_recall'] for r in results]),
            'rouge1': np.mean([r['metrics']['rouge1'] for r in results]),
            'rouge2': np.mean([r['metrics']['rouge2'] for r in results]),
            'rougeL': np.mean([r['metrics']['rougeL'] for r in results]),
        }
        
        # By question type
        by_type = defaultdict(list)
        for r in results:
            q_type = r.get('question_type', 'unknown')
            by_type[q_type].append(r)
        
        type_metrics = {}
        for q_type, type_results in by_type.items():
            type_metrics[q_type] = {
                'count': len(type_results),
                'mrr': np.mean([r['metrics']['mrr'] for r in type_results]),
                'ndcg_5': np.mean([r['metrics']['ndcg_5'] for r in type_results]),
                'bertscore_f1': np.mean([r['metrics']['bertscore_f1'] for r in type_results]),
            }
        
        # By difficulty
        by_difficulty = defaultdict(list)
        for r in results:
            difficulty = r.get('difficulty', 'unknown')
            by_difficulty[difficulty].append(r)
        
        difficulty_metrics = {}
        for difficulty, diff_results in by_difficulty.items():
            difficulty_metrics[difficulty] = {
                'count': len(diff_results),
                'mrr': np.mean([r['metrics']['mrr'] for r in diff_results]),
                'ndcg_5': np.mean([r['metrics']['ndcg_5'] for r in diff_results]),
                'bertscore_f1': np.mean([r['metrics']['bertscore_f1'] for r in diff_results]),
            }
        
        # Time statistics
        time_stats = {
            'avg_retrieval_time': np.mean([r['retrieval_time'] for r in results]),
            'avg_generation_time': np.mean([r['generation_time'] for r in results]),
            'avg_total_time': np.mean([r['total_time'] for r in results]),
        }
        
        return {
            'overall_metrics': metrics,
            'by_question_type': type_metrics,
            'by_difficulty': difficulty_metrics,
            'time_statistics': time_stats,
            'total_questions': len(results)
        }
    
    def generate_metric_explanations(self) -> Dict[str, str]:
        """
        Generate detailed explanations for each metric
        """
        return {
            'mrr': """
Mean Reciprocal Rank (MRR) - URL Level [MANDATORY METRIC]

WHY CHOSEN:
MRR is fundamental for evaluating retrieval quality in RAG systems. It measures
how quickly the system identifies the correct source document, which directly
impacts the quality of generated answers. Finding relevant sources early is
crucial for providing good context to the LLM.

CALCULATION:
1. For each query, retrieve top-K documents
2. Find the rank position of the first correct Wikipedia URL
3. Calculate reciprocal rank: RR = 1/rank (or 0 if not found)
4. Average across all queries: MRR = (1/N) * Σ(1/rank_i)

INTERPRETATION:
- 1.0: Perfect - correct URL always ranked first
- 0.7-1.0: Excellent retrieval
- 0.5-0.7: Good retrieval
- 0.3-0.5: Moderate retrieval
- <0.3: Poor retrieval, needs improvement

Higher MRR means better retrieval, leading to more relevant context
for answer generation.
            """,
            
            'bertscore': """
BERTScore F1 [CUSTOM METRIC 1]

WHY CHOSEN:
BERTScore evaluates semantic similarity between generated and ground truth answers
using contextual embeddings. Unlike lexical metrics, it understands meaning even
when wording differs. This is critical for RAG where the same information can be
expressed differently. It correlates well with human judgments of answer quality.

CALCULATION:
1. Compute BERT embeddings for each token in prediction and reference
2. Calculate cosine similarity matrix between all token pairs
3. Use greedy matching to find optimal token alignment
4. Compute:
   - Precision = avg max similarity for each predicted token
   - Recall = avg max similarity for each reference token
   - F1 = 2 * (P * R) / (P + R)

INTERPRETATION:
- >0.9: Excellent - answer semantically equivalent to ground truth
- 0.8-0.9: Good - minor semantic differences
- 0.7-0.8: Moderate - some information missing/incorrect
- 0.6-0.7: Fair - significant gaps in content
- <0.6: Poor - answer diverges from expected content

Precision measures relevance (no hallucinations), Recall measures completeness
(no missing information), F1 balances both.
            """,
            
            'ndcg': """
Normalized Discounted Cumulative Gain @ 5 (NDCG@5) [CUSTOM METRIC 2]

WHY CHOSEN:
NDCG@K evaluates ranking quality by considering both relevance and position.
It's crucial for RAG because position matters - documents higher in the ranking
provide context to the LLM first. Good ranking means relevant information
appears early, improving answer quality. Unlike MRR, NDCG considers all
relevant documents, not just the first one.

CALCULATION:
1. Create binary relevance labels: rel_i = 1 if URL_i is relevant, 0 otherwise
2. Calculate DCG@5:
   DCG@5 = Σ(i=1 to 5) [rel_i / log2(i+1)]
   (Logarithmic discount - later positions contribute less)
3. Calculate ideal DCG (IDCG) with perfect ranking
4. Normalize: NDCG@5 = DCG@5 / IDCG@5

INTERPRETATION:
- 1.0: Perfect ranking - all relevant docs at top
- 0.8-1.0: Excellent ranking quality
- 0.6-0.8: Good - most relevant docs in top positions
- 0.4-0.6: Moderate - relevant docs scattered
- <0.4: Poor - relevant docs buried or missing

Higher NDCG means the LLM receives better context earlier, leading to
more accurate and complete answers.
            """
        }


def main():
    """Demo evaluation"""
    evaluator = RAGEvaluator()
    
    # Print metric explanations
    explanations = evaluator.generate_metric_explanations()
    print("="*80)
    print("EVALUATION METRICS DOCUMENTATION")
    print("="*80)
    
    for metric_name, explanation in explanations.items():
        print(f"\n{explanation}")
        print("-"*80)

    def llm_judge_answer(
        self,
        query: str,
        generated_answer: str,
        ground_truth: str,
        retrieved_context: List[str] = None
    ) -> Dict[str, float]:
        """
        LLM-as-Judge evaluation using heuristic-based scoring
        
        Evaluates:
        1. Factual Accuracy - How correct is the answer?
        2. Completeness - Does it cover all key points?
        3. Relevance - Does it answer the question?
        4. Coherence - Is it well-structured?
        5. Hallucination - Contains unsupported facts?
        
        Args:
            query: User question
            generated_answer: Model's answer
            ground_truth: Expected answer
            retrieved_context: Context chunks used (optional)
        
        Returns:
            Dictionary with scores (0-1) for each dimension
        """
        if not generated_answer or not ground_truth:
            return {
                'factual_accuracy': 0.0,
                'completeness': 0.0,
                'relevance': 0.0,
                'coherence': 0.0,
                'hallucination_score': 1.0,  # High = bad
                'overall': 0.0
            }
        
        # Normalize texts
        gen_lower = generated_answer.lower()
        gt_lower = ground_truth.lower()
        query_lower = query.lower()
        
        # 1. FACTUAL ACCURACY - Use ROUGE-L as proxy
        rouge_scores = self.rouge_scorer.score(ground_truth, generated_answer)
        factual_accuracy = rouge_scores['rougeL'].fmeasure
        
        # 2. COMPLETENESS - Keyword coverage
        import re
        gt_words = set(re.findall(r'\w+', gt_lower))
        gen_words = set(re.findall(r'\w+', gen_lower))
        
        # Key terms in ground truth
        gt_keywords = {w for w in gt_words if len(w) > 4}  # Longer words
        if gt_keywords:
            covered_keywords = gt_keywords.intersection(gen_words)
            completeness = len(covered_keywords) / len(gt_keywords)
        else:
            completeness = 0.5
        
        # 3. RELEVANCE - Query term coverage in answer
        query_words = set(re.findall(r'\w+', query_lower))
        query_keywords = {w for w in query_words if len(w) > 3}
        if query_keywords:
            covered_query = query_keywords.intersection(gen_words)
            relevance = len(covered_query) / len(query_keywords)
        else:
            relevance = 0.5
        
        # Boost if answer length is reasonable
        if 20 < len(generated_answer.split()) < 200:
            relevance = min(1.0, relevance + 0.1)
        
        # 4. COHERENCE - Structural quality
        coherence_score = 0.5
        
        # Has complete sentences
        if generated_answer.strip().endswith(('.', '!', '?')):
            coherence_score += 0.15
        
        # Not too short or too long
        word_count = len(generated_answer.split())
        if 15 <= word_count <= 150:
            coherence_score += 0.15
        
        # No repetition
        unique_ratio = len(set(gen_words)) / max(len(gen_words.split()), 1)
        if unique_ratio > 0.7:
            coherence_score += 0.1
        
        # Has proper capitalization
        if generated_answer[0].isupper():
            coherence_score += 0.1
        
        coherence_score = min(1.0, coherence_score)
        
        # 5. HALLUCINATION DETECTION
        if retrieved_context:
            # Check if generated content is supported by context
            context_text = ' '.join(retrieved_context).lower()
            context_words = set(re.findall(r'\w+', context_text))
            
            # Important words in answer not in context = potential hallucination
            important_gen_words = {w for w in gen_words if len(w) > 5}
            if important_gen_words:
                unsupported = important_gen_words - context_words - query_keywords
                hallucination_rate = len(unsupported) / len(important_gen_words)
            else:
                hallucination_rate = 0.0
        else:
            # Without context, use ground truth as reference
            important_gen_words = {w for w in gen_words if len(w) > 5}
            if important_gen_words:
                unsupported = important_gen_words - gt_keywords - query_keywords
                hallucination_rate = len(unsupported) / len(important_gen_words)
            else:
                hallucination_rate = 0.0
        
        hallucination_score = 1.0 - hallucination_rate  # High = good (no hallucinations)
        
        # OVERALL SCORE (weighted average)
        overall = (
            0.30 * factual_accuracy +
            0.25 * completeness +
            0.20 * relevance +
            0.15 * coherence_score +
            0.10 * hallucination_score
        )
        
        return {
            'factual_accuracy': round(factual_accuracy, 4),
            'completeness': round(completeness, 4),
            'relevance': round(relevance, 4),
            'coherence': round(coherence_score, 4),
            'hallucination_score': round(hallucination_score, 4),
            'overall': round(overall, 4),
            'explanation': self._generate_judge_explanation(
                factual_accuracy, completeness, relevance, 
                coherence_score, hallucination_score
            )
        }
    
    def _generate_judge_explanation(
        self, 
        factual: float, 
        complete: float, 
        relevant: float, 
        coherent: float, 
        halluc: float
    ) -> str:
        """Generate human-readable explanation of scores"""
        parts = []
        
        if factual >= 0.8:
            parts.append("✓ Factually accurate")
        elif factual >= 0.6:
            parts.append("~ Mostly accurate")
        else:
            parts.append("✗ Low factual accuracy")
        
        if complete >= 0.7:
            parts.append("✓ Complete answer")
        elif complete >= 0.5:
            parts.append("~ Partially complete")
        else:
            parts.append("✗ Missing key information")
        
        if relevant >= 0.7:
            parts.append("✓ Highly relevant")
        elif relevant >= 0.5:
            parts.append("~ Somewhat relevant")
        else:
            parts.append("✗ Low relevance")
        
        if coherent >= 0.7:
            parts.append("✓ Well-structured")
        else:
            parts.append("~ Could be more coherent")
        
        if halluc >= 0.8:
            parts.append("✓ No hallucinations")
        elif halluc >= 0.6:
            parts.append("~ Minor unsupported claims")
        else:
            parts.append("✗ Contains hallucinations")
        
        return " | ".join(parts)


if __name__ == "__main__":
    main()
