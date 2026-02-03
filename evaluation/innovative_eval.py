"""
Innovative Evaluation Features
- Ablation studies
- Error analysis
- LLM-as-judge
- Adversarial testing
- Confidence calibration
"""

import json
import random
import numpy as np
from typing import List, Dict
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# For LLM-as-judge
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class InnovativeEvaluator:
    """Advanced evaluation techniques"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.judge_model = None
        self.judge_tokenizer = None
    
    def ablation_study(self, rag_system, questions: List[Dict], 
                      output_dir: str = "reports/ablation") -> Dict:
        """
        Compare dense-only, sparse-only, and hybrid performance
        """
        print("\n" + "="*60)
        print("ABLATION STUDY: Comparing Retrieval Methods")
        print("="*60)
        
        methods = ['dense', 'sparse', 'hybrid']
        results_by_method = {}
        
        from .metrics import RAGEvaluator
        evaluator = RAGEvaluator()
        
        for method in methods:
            print(f"\nEvaluating {method} method...")
            method_results = []
            
            for question in questions[:20]:  # Sample for ablation
                # Get RAG response
                response = rag_system.query(question['question'], method=method)
                
                # Evaluate
                eval_result = evaluator.evaluate_single_query(question, response)
                method_results.append(eval_result)
            
            # Aggregate
            aggregated = evaluator.aggregate_results(method_results)
            results_by_method[method] = aggregated
            
            print(f"  {method.upper()}: MRR={aggregated['overall_metrics']['mrr']:.3f}, "
                  f"NDCG@5={aggregated['overall_metrics']['ndcg_5']:.3f}, "
                  f"BERTScore={aggregated['overall_metrics']['bertscore_f1']:.3f}")
        
        # Create comparison visualization
        self._visualize_ablation(results_by_method, output_dir)
        
        return results_by_method
    
    def _visualize_ablation(self, results: Dict, output_dir: str):
        """Create ablation study visualizations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        methods = list(results.keys())
        metrics = ['mrr', 'ndcg_5', 'bertscore_f1', 'precision_5']
        
        data = []
        for method in methods:
            for metric in metrics:
                value = results[method]['overall_metrics'].get(metric, 0)
                data.append({
                    'Method': method,
                    'Metric': metric.upper().replace('_', '@'),
                    'Score': value
                })
        
        # Create grouped bar chart
        import pandas as pd
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        pivot_df = df.pivot(index='Metric', columns='Method', values='Score')
        pivot_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Ablation Study: Comparison of Retrieval Methods', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.legend(title='Method', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ablation_comparison.png", dpi=300)
        plt.close()
        
        print(f"✓ Saved ablation study visualization to {output_dir}/ablation_comparison.png")
    
    def error_analysis(self, evaluation_results: List[Dict], 
                      output_dir: str = "reports/errors") -> Dict:
        """
        Categorize and analyze failures
        """
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Categorize by performance
        failures = []
        partial_successes = []
        successes = []
        
        for result in evaluation_results:
            mrr = result['metrics']['mrr']
            bert_f1 = result['metrics']['bertscore_f1']
            
            if mrr == 0 or bert_f1 < 0.5:
                failures.append(result)
            elif mrr < 0.5 or bert_f1 < 0.7:
                partial_successes.append(result)
            else:
                successes.append(result)
        
        print(f"\nSuccess: {len(successes)} ({len(successes)/len(evaluation_results)*100:.1f}%)")
        print(f"Partial Success: {len(partial_successes)} ({len(partial_successes)/len(evaluation_results)*100:.1f}%)")
        print(f"Failures: {len(failures)} ({len(failures)/len(evaluation_results)*100:.1f}%)")
        
        # Analyze failure patterns by question type
        failure_by_type = defaultdict(list)
        for fail in failures:
            q_type = fail.get('question_type', 'unknown')
            failure_by_type[q_type].append(fail)
        
        print("\nFailures by Question Type:")
        for q_type, fails in failure_by_type.items():
            print(f"  {q_type}: {len(fails)}")
        
        # Error categories
        error_categories = {
            'retrieval_failure': [],  # MRR = 0
            'generation_failure': [],  # MRR > 0 but BERTScore low
            'both_failure': []  # Both failed
        }
        
        for fail in failures:
            if fail['metrics']['mrr'] == 0 and fail['metrics']['bertscore_f1'] < 0.5:
                error_categories['both_failure'].append(fail)
            elif fail['metrics']['mrr'] == 0:
                error_categories['retrieval_failure'].append(fail)
            else:
                error_categories['generation_failure'].append(fail)
        
        print("\nError Categories:")
        for category, errors in error_categories.items():
            print(f"  {category}: {len(errors)}")
        
        # Save failure examples
        failure_report = {
            'summary': {
                'total_evaluated': len(evaluation_results),
                'successes': len(successes),
                'partial_successes': len(partial_successes),
                'failures': len(failures),
                'failure_rate': len(failures) / len(evaluation_results)
            },
            'failure_by_type': {k: len(v) for k, v in failure_by_type.items()},
            'error_categories': {k: len(v) for k, v in error_categories.items()},
            'failure_examples': failures[:10]  # First 10 failures
        }
        
        with open(f"{output_dir}/error_analysis.json", 'w') as f:
            json.dump(failure_report, f, indent=2)
        
        # Visualize
        self._visualize_errors(failure_report, output_dir)
        
        return failure_report
    
    def _visualize_errors(self, report: Dict, output_dir: str):
        """Create error analysis visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of success/partial/failure
        labels = ['Success', 'Partial Success', 'Failures']
        sizes = [
            report['summary']['successes'],
            report['summary']['partial_successes'],
            report['summary']['failures']
        ]
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Overall Performance Distribution', fontweight='bold')
        
        # Bar chart of error categories
        categories = list(report['error_categories'].keys())
        counts = list(report['error_categories'].values())
        
        axes[1].bar(categories, counts, color=['#ff6b6b', '#feca57', '#ee5a6f'])
        axes[1].set_title('Error Category Breakdown', fontweight='bold')
        axes[1].set_ylabel('Count')
        axes[1].set_xlabel('Error Type')
        axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_distribution.png", dpi=300)
        plt.close()
        
        print(f"✓ Saved error analysis to {output_dir}/")
    
    def llm_as_judge(self, predictions: List[str], references: List[str], 
                     questions: List[str]) -> List[Dict]:
        """
        Use LLM to judge answer quality
        Evaluates: factual accuracy, completeness, relevance, coherence
        """
        print("\n" + "="*60)
        print("LLM-AS-JUDGE EVALUATION")
        print("="*60)
        
        # Load judge model
        if self.judge_model is None:
            print("Loading judge model (Flan-T5)...")
            self.judge_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.judge_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            
            if torch.cuda.is_available():
                self.judge_model = self.judge_model.cuda()
        
        judgments = []
        
        for pred, ref, question in zip(predictions[:10], references[:10], questions[:10]):
            # Prepare judgment prompt
            prompt = f"""Evaluate the following answer on a scale of 1-5 for:
1. Factual Accuracy (is it correct?)
2. Completeness (does it fully answer the question?)
3. Relevance (is it on-topic?)
4. Coherence (is it well-structured?)

Question: {question}
Reference Answer: {ref}
Generated Answer: {pred}

Provide scores and brief explanation:"""
            
            # Tokenize
            inputs = self.judge_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate judgment
            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
            
            judgment = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            judgments.append({
                'question': question,
                'prediction': pred,
                'reference': ref,
                'judgment': judgment
            })
        
        print(f"✓ Generated {len(judgments)} LLM judgments")
        return judgments
    
    def adversarial_testing(self, rag_system, base_questions: List[Dict]) -> Dict:
        """
        Test with challenging variations
        """
        print("\n" + "="*60)
        print("ADVERSARIAL TESTING")
        print("="*60)
        
        from .metrics import RAGEvaluator
        evaluator = RAGEvaluator()
        
        adversarial_results = {
            'negated': [],
            'paraphrased': [],
            'ambiguous': []
        }
        
        # Test negated questions
        print("\n1. Testing with negated questions...")
        for q in base_questions[:5]:
            original_question = q['question']
            
            # Create negated version
            if '?' in original_question:
                negated = original_question.replace('?', '').replace('What is', 'What is not')
                if 'What is not' not in negated:
                    negated = f"What is NOT {original_question}"
                
                response = rag_system.query(negated, method='hybrid')
                eval_result = evaluator.evaluate_single_query(q, response)
                adversarial_results['negated'].append({
                    'original': original_question,
                    'adversarial': negated,
                    'mrr': eval_result['metrics']['mrr']
                })
        
        # Test paraphrased questions
        print("2. Testing with paraphrased questions...")
        paraphrase_templates = [
            "Can you explain {}?",
            "Tell me about {}.",
            "I want to know about {}.",
            "Give me information on {}."
        ]
        
        for q in base_questions[:5]:
            original = q['question']
            # Extract topic
            topic = q.get('source_title', 'this topic')
            
            paraphrased = random.choice(paraphrase_templates).format(topic)
            response = rag_system.query(paraphrased, method='hybrid')
            eval_result = evaluator.evaluate_single_query(q, response)
            adversarial_results['paraphrased'].append({
                'original': original,
                'paraphrased': paraphrased,
                'mrr': eval_result['metrics']['mrr']
            })
        
        print("✓ Adversarial testing complete")
        return adversarial_results
    
    def confidence_calibration(self, evaluation_results: List[Dict]) -> Dict:
        """
        Analyze confidence calibration
        High retrieval scores should correlate with correct answers
        """
        print("\n" + "="*60)
        print("CONFIDENCE CALIBRATION ANALYSIS")
        print("="*60)
        
        # Extract confidence (RRF score) and correctness (MRR)
        confidences = []
        correctness = []
        
        for result in evaluation_results:
            # Use RRF score as confidence proxy
            if result.get('retrieved_urls'):
                # Confidence based on top retrieval score
                confidence = result['metrics'].get('ndcg_5', 0)
                correct = 1 if result['metrics']['mrr'] > 0 else 0
                
                confidences.append(confidence)
                correctness.append(correct)
        
        # Calculate calibration
        bins = np.linspace(0, 1, 11)
        bin_accuracy = []
        bin_confidence = []
        
        for i in range(len(bins) - 1):
            mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
            if mask.sum() > 0:
                bin_accuracy.append(np.array(correctness)[mask].mean())
                bin_confidence.append(bins[i])
        
        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(np.array(bin_accuracy) - np.array(bin_confidence)))
        
        print(f"Expected Calibration Error: {ece:.3f}")
        print("(Lower is better, 0 = perfect calibration)")
        
        return {
            'ece': ece,
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_confidence
        }


def main():
    """Demo innovative evaluation"""
    print("Innovative Evaluation Features Module")
    print("This module provides advanced evaluation capabilities")


if __name__ == "__main__":
    main()
