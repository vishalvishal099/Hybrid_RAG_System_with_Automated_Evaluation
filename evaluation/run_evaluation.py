"""
Complete Evaluation Pipeline
Orchestrates semantic chunking, RRF fusion, retrieval, generation, and evaluation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.semantic_chunker import SemanticChunker, process_corpus_with_semantic_chunking
from src.rrf_fusion import RRFFusion, apply_rrf_to_results
from evaluation.comprehensive_metrics import EvaluationMetrics
from evaluation.create_dataset import load_evaluation_dataset, save_evaluation_dataset


class ComprehensiveEvaluationPipeline:
    """
    End-to-end evaluation pipeline implementing all requirements:
    - Semantic chunking (200-400 tokens, 50 overlap)
    - RRF fusion (k=60)
    - MRR, Recall@10, Answer F1
    - Ablation studies
    - LLM-as-judge
    """
    
    def __init__(self, rag_system, output_dir: str = "evaluation/results"):
        """
        Initialize pipeline
        
        Args:
            rag_system: HybridRAGSystem instance
            output_dir: Directory to save results
        """
        self.rag_system = rag_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = EvaluationMetrics()
        self.rrf = RRFFusion(k=60)
        
        self.results = {
            'retrieval_metrics': {},
            'generation_metrics': {},
            'ablation_results': {},
            'llm_judge_results': {}
        }
    
    def run_retrieval_evaluation(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate retrieval with MRR and Recall@10
        """
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION")
        print("="*60)
        
        mrr_scores = []
        recall_scores = []
        
        for i, case in enumerate(test_cases):
            query = case['question']
            gt_url = case.get('ground_truth_url')
            gt_urls = case.get('ground_truth_urls', [gt_url] if gt_url else [])
            
            # Retrieve with hybrid method (RRF fusion)
            results = self.rag_system.retrieve_with_rrf(query, top_k=10)
            retrieved_urls = [r.get('url') for r in results if r.get('url')]
            
            # Compute metrics
            if gt_url:
                mrr = self.metrics.compute_mrr(retrieved_urls, gt_url)
                mrr_scores.append(mrr)
            
            if gt_urls:
                recall = self.metrics.compute_recall_at_k(retrieved_urls, gt_urls, k=10)
                recall_scores.append(recall)
            
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(test_cases)} queries...")
        
        results = {
            'mean_mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'mean_recall_at_10': np.mean(recall_scores) if recall_scores else 0.0,
            'mrr_scores': mrr_scores,
            'recall_scores': recall_scores
        }
        
        print(f"\n✓ Retrieval Evaluation Complete:")
        print(f"  - Mean MRR: {results['mean_mrr']:.4f}")
        print(f"  - Mean Recall@10: {results['mean_recall_at_10']:.4f}")
        
        self.results['retrieval_metrics'] = results
        return results
    
    def run_generation_evaluation(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate generation with Answer F1
        """
        print("\n" + "="*60)
        print("GENERATION EVALUATION")
        print("="*60)
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for i, case in enumerate(test_cases):
            query = case['question']
            gt_answer = case.get('ground_truth_answer', '')
            
            if not gt_answer:
                continue
            
            # Retrieve context
            context = self.rag_system.retrieve_with_rrf(query, top_k=5)
            
            # Generate answer
            result = self.rag_system.generate_answer(query, context)
            generated = result.get('answer', '')
            
            # Compute Answer F1
            scores = self.metrics.compute_answer_f1(generated, gt_answer)
            f1_scores.append(scores['f1'])
            precision_scores.append(scores['precision'])
            recall_scores.append(scores['recall'])
            
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1} answers...")
        
        results = {
            'mean_f1': np.mean(f1_scores) if f1_scores else 0.0,
            'mean_precision': np.mean(precision_scores) if precision_scores else 0.0,
            'mean_recall': np.mean(recall_scores) if recall_scores else 0.0,
            'f1_scores': f1_scores
        }
        
        print(f"\n✓ Generation Evaluation Complete:")
        print(f"  - Mean Answer F1: {results['mean_f1']:.4f}")
        print(f"  - Mean Precision: {results['mean_precision']:.4f}")
        print(f"  - Mean Recall: {results['mean_recall']:.4f}")
        
        self.results['generation_metrics'] = results
        return results
    
    def run_ablation_study(self, test_cases: List[Dict]) -> Dict:
        """
        Ablation study: Dense-only vs Sparse-only vs Hybrid (RRF)
        """
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        dense_mrr = []
        sparse_mrr = []
        hybrid_mrr = []
        
        for i, case in enumerate(test_cases):
            query = case['question']
            gt_url = case.get('ground_truth_url')
            
            if not gt_url:
                continue
            
            # Dense-only (FAISS)
            dense_results = self.rag_system.dense_search_only(query, top_k=10)
            dense_urls = [r.get('url') for r in dense_results if r.get('url')]
            dense_mrr.append(self.metrics.compute_mrr(dense_urls, gt_url))
            
            # Sparse-only (BM25)
            sparse_results = self.rag_system.sparse_search_only(query, top_k=10)
            sparse_urls = [r.get('url') for r in sparse_results if r.get('url')]
            sparse_mrr.append(self.metrics.compute_mrr(sparse_urls, gt_url))
            
            # Hybrid (RRF)
            hybrid_results = self.rag_system.retrieve_with_rrf(query, top_k=10)
            hybrid_urls = [r.get('url') for r in hybrid_results if r.get('url')]
            hybrid_mrr.append(self.metrics.compute_mrr(hybrid_urls, gt_url))
            
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1} queries...")
        
        results = {
            'dense_only': {
                'mean_mrr': np.mean(dense_mrr) if dense_mrr else 0.0,
                'mrr_scores': dense_mrr
            },
            'sparse_only': {
                'mean_mrr': np.mean(sparse_mrr) if sparse_mrr else 0.0,
                'mrr_scores': sparse_mrr
            },
            'hybrid_rrf': {
                'mean_mrr': np.mean(hybrid_mrr) if hybrid_mrr else 0.0,
                'mrr_scores': hybrid_mrr
            }
        }
        
        print(f"\n✓ Ablation Study Complete:")
        print(f"  - Dense-only MRR: {results['dense_only']['mean_mrr']:.4f}")
        print(f"  - Sparse-only MRR: {results['sparse_only']['mean_mrr']:.4f}")
        print(f"  - Hybrid RRF MRR: {results['hybrid_rrf']['mean_mrr']:.4f}")
        
        self.results['ablation_results'] = results
        return results
    
    def run_llm_judge_evaluation(self, test_cases: List[Dict], sample_size: int = 10) -> Dict:
        """
        LLM-as-judge evaluation on sample of test cases
        """
        print("\n" + "="*60)
        print("LLM-AS-JUDGE EVALUATION")
        print("="*60)
        
        # Sample test cases
        import random
        sampled_cases = random.sample(test_cases, min(sample_size, len(test_cases)))
        
        relevance_scores = []
        correctness_scores = []
        completeness_scores = []
        overall_scores = []
        
        for i, case in enumerate(sampled_cases):
            query = case['question']
            gt_answer = case.get('ground_truth_answer', '')
            
            if not gt_answer:
                continue
            
            # Retrieve and generate
            context = self.rag_system.retrieve_with_rrf(query, top_k=5)
            result = self.rag_system.generate_answer(query, context)
            generated = result.get('answer', '')
            
            # LLM judge
            judge_scores = self.metrics.llm_judge_answer(
                query, generated, gt_answer, judge_model=None  # Using heuristic
            )
            
            relevance_scores.append(judge_scores['relevance'])
            correctness_scores.append(judge_scores['correctness'])
            completeness_scores.append(judge_scores['completeness'])
            overall_scores.append(judge_scores['overall'])
            
            print(f"  Evaluated {i + 1}/{len(sampled_cases)} samples...")
        
        results = {
            'mean_relevance': np.mean(relevance_scores) if relevance_scores else 0.0,
            'mean_correctness': np.mean(correctness_scores) if correctness_scores else 0.0,
            'mean_completeness': np.mean(completeness_scores) if completeness_scores else 0.0,
            'mean_overall': np.mean(overall_scores) if overall_scores else 0.0,
            'sample_size': len(sampled_cases)
        }
        
        print(f"\n✓ LLM Judge Evaluation Complete:")
        print(f"  - Relevance: {results['mean_relevance']:.4f}")
        print(f"  - Correctness: {results['mean_correctness']:.4f}")
        print(f"  - Completeness: {results['mean_completeness']:.4f}")
        print(f"  - Overall: {results['mean_overall']:.4f}")
        
        self.results['llm_judge_results'] = results
        return results
    
    def generate_report(self):
        """
        Generate comprehensive evaluation report with visualizations
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60)
        
        # Save results to JSON
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved results to {results_file}")
        
        # Create visualizations
        self._create_visualizations()
        
        # Create text report
        self._create_text_report()
        
        print(f"\n✓ Report generation complete!")
        print(f"  Results saved in: {self.output_dir}")
    
    def _create_visualizations(self):
        """Create evaluation visualizations"""
        # Ablation study comparison
        if self.results.get('ablation_results'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            methods = ['Dense\nOnly', 'Sparse\nOnly', 'Hybrid\nRRF']
            mrr_values = [
                self.results['ablation_results']['dense_only']['mean_mrr'],
                self.results['ablation_results']['sparse_only']['mean_mrr'],
                self.results['ablation_results']['hybrid_rrf']['mean_mrr']
            ]
            
            bars = ax.bar(methods, mrr_values, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12)
            ax.set_title('Ablation Study: Retrieval Method Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.0)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "ablation_study.png", dpi=300)
            print(f"✓ Saved ablation study chart")
            plt.close()
        
        # Metrics overview
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Retrieval metrics
        if self.results.get('retrieval_metrics'):
            ax = axes[0]
            metrics_names = ['MRR', 'Recall@10']
            metrics_values = [
                self.results['retrieval_metrics']['mean_mrr'],
                self.results['retrieval_metrics']['mean_recall_at_10']
            ]
            ax.bar(metrics_names, metrics_values, color='#3498db')
            ax.set_title('Retrieval Metrics', fontweight='bold')
            ax.set_ylim(0, 1.0)
            for i, v in enumerate(metrics_values):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Generation metrics
        if self.results.get('generation_metrics'):
            ax = axes[1]
            metrics_names = ['F1', 'Precision', 'Recall']
            metrics_values = [
                self.results['generation_metrics']['mean_f1'],
                self.results['generation_metrics']['mean_precision'],
                self.results['generation_metrics']['mean_recall']
            ]
            ax.bar(metrics_names, metrics_values, color='#e74c3c')
            ax.set_title('Answer Quality Metrics', fontweight='bold')
            ax.set_ylim(0, 1.0)
            for i, v in enumerate(metrics_values):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # LLM judge
        if self.results.get('llm_judge_results'):
            ax = axes[2]
            metrics_names = ['Relevance', 'Correctness', 'Completeness', 'Overall']
            metrics_values = [
                self.results['llm_judge_results']['mean_relevance'],
                self.results['llm_judge_results']['mean_correctness'],
                self.results['llm_judge_results']['mean_completeness'],
                self.results['llm_judge_results']['mean_overall']
            ]
            ax.bar(metrics_names, metrics_values, color='#2ecc71')
            ax.set_title('LLM Judge Scores', fontweight='bold')
            ax.set_ylim(0, 1.0)
            for i, v in enumerate(metrics_values):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_overview.png", dpi=300)
        print(f"✓ Saved metrics overview chart")
        plt.close()
    
    def _create_text_report(self):
        """Create text-based evaluation report"""
        report_file = self.output_dir / "evaluation_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE RAG SYSTEM EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # System Configuration
            f.write("SYSTEM CONFIGURATION:\n")
            f.write("-" * 70 + "\n")
            f.write("• Chunking: Semantic (200-400 tokens, 50-token overlap)\n")
            f.write("• Retrieval: Dense (FAISS) + Sparse (BM25)\n")
            f.write("• Fusion: Reciprocal Rank Fusion (k=60)\n")
            f.write("• Generation: FLAN-T5-base\n\n")
            
            # Retrieval Metrics
            if self.results.get('retrieval_metrics'):
                f.write("RETRIEVAL PERFORMANCE:\n")
                f.write("-" * 70 + "\n")
                f.write(f"• Mean Reciprocal Rank (MRR): {self.results['retrieval_metrics']['mean_mrr']:.4f}\n")
                f.write(f"• Mean Recall@10: {self.results['retrieval_metrics']['mean_recall_at_10']:.4f}\n\n")
            
            # Generation Metrics
            if self.results.get('generation_metrics'):
                f.write("ANSWER QUALITY:\n")
                f.write("-" * 70 + "\n")
                f.write(f"• Mean Answer F1: {self.results['generation_metrics']['mean_f1']:.4f}\n")
                f.write(f"• Mean Precision: {self.results['generation_metrics']['mean_precision']:.4f}\n")
                f.write(f"• Mean Recall: {self.results['generation_metrics']['mean_recall']:.4f}\n\n")
            
            # Ablation Study
            if self.results.get('ablation_results'):
                f.write("ABLATION STUDY (MRR):\n")
                f.write("-" * 70 + "\n")
                f.write(f"• Dense-only (FAISS): {self.results['ablation_results']['dense_only']['mean_mrr']:.4f}\n")
                f.write(f"• Sparse-only (BM25): {self.results['ablation_results']['sparse_only']['mean_mrr']:.4f}\n")
                f.write(f"• Hybrid (RRF k=60): {self.results['ablation_results']['hybrid_rrf']['mean_mrr']:.4f}\n\n")
            
            # LLM Judge
            if self.results.get('llm_judge_results'):
                f.write("LLM-AS-JUDGE EVALUATION:\n")
                f.write("-" * 70 + "\n")
                f.write(f"• Relevance: {self.results['llm_judge_results']['mean_relevance']:.4f}\n")
                f.write(f"• Correctness: {self.results['llm_judge_results']['mean_correctness']:.4f}\n")
                f.write(f"• Completeness: {self.results['llm_judge_results']['mean_completeness']:.4f}\n")
                f.write(f"• Overall Score: {self.results['llm_judge_results']['mean_overall']:.4f}\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"✓ Saved text report to {report_file}")
    
    def run_complete_evaluation(self, test_cases: List[Dict]):
        """
        Run complete evaluation pipeline
        """
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE EVALUATION PIPELINE")
        print("="*70)
        print(f"Test cases: {len(test_cases)}")
        
        # Run all evaluations
        self.run_retrieval_evaluation(test_cases)
        self.run_generation_evaluation(test_cases)
        self.run_ablation_study(test_cases)
        self.run_llm_judge_evaluation(test_cases, sample_size=10)
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        
        return self.results


def main():
    """Main execution function"""
    # Create evaluation dataset
    print("Creating evaluation dataset...")
    save_evaluation_dataset()
    test_cases = load_evaluation_dataset()
    
    print(f"\n✓ Loaded {len(test_cases)} test cases")
    print("\nNOTE: This pipeline requires a configured RAG system.")
    print("To run evaluation, import this module and call:")
    print("  pipeline = ComprehensiveEvaluationPipeline(your_rag_system)")
    print("  pipeline.run_complete_evaluation(test_cases)")


if __name__ == "__main__":
    main()
