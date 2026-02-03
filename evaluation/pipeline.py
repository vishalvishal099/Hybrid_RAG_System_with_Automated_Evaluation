"""
Automated Evaluation Pipeline
Single command to run full evaluation and generate reports
"""

import json
import sys
import time
from pathlib import Path
import yaml
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_system import HybridRAGSystem
from evaluation.metrics import RAGEvaluator
from evaluation.innovative_eval import InnovativeEvaluator


class EvaluationPipeline:
    """Automated evaluation pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rag_system = None
        self.evaluator = RAGEvaluator(config_path)
        self.innovative_evaluator = InnovativeEvaluator(config_path)
        
        # Results
        self.evaluation_results = []
        self.aggregated_results = {}
        self.ablation_results = {}
        self.error_analysis = {}
        
        print("✓ Evaluation Pipeline initialized")
    
    def load_questions(self, questions_path: str = None) -> list:
        """Load evaluation questions"""
        if questions_path is None:
            questions_path = self.config['paths']['questions_dataset']
        
        print(f"\nLoading questions from {questions_path}...")
        with open(questions_path, 'r') as f:
            data = json.load(f)
        
        questions = data['questions']
        print(f"✓ Loaded {len(questions)} questions")
        print(f"  Question types: {data['metadata']['question_types']}")
        
        return questions
    
    def initialize_rag_system(self):
        """Initialize and load RAG system"""
        print("\nInitializing RAG system...")
        self.rag_system = HybridRAGSystem()
        self.rag_system.load_corpus()
        self.rag_system.load_indexes()
        print("✓ RAG system ready")
    
    def run_evaluation(self, questions: list, method: str = "hybrid"):
        """Run evaluation on all questions"""
        print(f"\n{'='*60}")
        print(f"RUNNING EVALUATION ({method.upper()} METHOD)")
        print(f"{'='*60}")
        
        from tqdm import tqdm
        
        self.evaluation_results = []
        
        for question in tqdm(questions, desc="Evaluating questions"):
            try:
                # Get RAG response
                response = self.rag_system.query(question['question'], method=method)
                
                # Evaluate
                eval_result = self.evaluator.evaluate_single_query(question, response)
                self.evaluation_results.append(eval_result)
                
            except Exception as e:
                print(f"\nError evaluating question {question['question_id']}: {e}")
                continue
        
        # Aggregate results
        self.aggregated_results = self.evaluator.aggregate_results(self.evaluation_results)
        
        print(f"\n✓ Evaluation complete!")
        print(f"\nOVERALL RESULTS:")
        print(f"  MRR:           {self.aggregated_results['overall_metrics']['mrr']:.4f}")
        print(f"  NDCG@5:        {self.aggregated_results['overall_metrics']['ndcg_5']:.4f}")
        print(f"  BERTScore F1:  {self.aggregated_results['overall_metrics']['bertscore_f1']:.4f}")
        print(f"  Precision@5:   {self.aggregated_results['overall_metrics']['precision_5']:.4f}")
        print(f"  Recall@5:      {self.aggregated_results['overall_metrics']['recall_5']:.4f}")
        print(f"  ROUGE-L:       {self.aggregated_results['overall_metrics']['rougeL']:.4f}")
    
    def run_ablation_study(self, questions: list):
        """Run ablation study"""
        print(f"\n{'='*60}")
        print("RUNNING ABLATION STUDY")
        print(f"{'='*60}")
        
        self.ablation_results = self.innovative_evaluator.ablation_study(
            self.rag_system,
            questions,
            output_dir="reports/ablation"
        )
    
    def run_error_analysis(self):
        """Run error analysis"""
        self.error_analysis = self.innovative_evaluator.error_analysis(
            self.evaluation_results,
            output_dir="reports/errors"
        )
    
    def generate_visualizations(self, output_dir: str = "reports/visualizations"):
        """Generate all visualizations"""
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Metrics comparison
        self._plot_metrics_comparison(output_dir)
        
        # 2. Performance by question type
        self._plot_performance_by_type(output_dir)
        
        # 3. Performance by difficulty
        self._plot_performance_by_difficulty(output_dir)
        
        # 4. Time distribution
        self._plot_time_distribution(output_dir)
        
        # 5. Score distributions
        self._plot_score_distributions(output_dir)
        
        print(f"✓ Visualizations saved to {output_dir}/")
    
    def _plot_metrics_comparison(self, output_dir: str):
        """Plot main metrics comparison"""
        metrics = ['mrr', 'ndcg_5', 'bertscore_f1', 'precision_5', 'recall_5']
        values = [self.aggregated_results['overall_metrics'][m] for m in metrics]
        labels = ['MRR', 'NDCG@5', 'BERTScore F1', 'Precision@5', 'Recall@5']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
        plt.ylim(0, 1.0)
        plt.ylabel('Score', fontsize=12)
        plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300)
        plt.close()
    
    def _plot_performance_by_type(self, output_dir: str):
        """Plot performance by question type"""
        by_type = self.aggregated_results['by_question_type']
        
        types = list(by_type.keys())
        mrr_scores = [by_type[t]['mrr'] for t in types]
        ndcg_scores = [by_type[t]['ndcg_5'] for t in types]
        bert_scores = [by_type[t]['bertscore_f1'] for t in types]
        
        x = range(len(types))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar([i - width for i in x], mrr_scores, width, label='MRR', color='#3498db')
        plt.bar(x, ndcg_scores, width, label='NDCG@5', color='#2ecc71')
        plt.bar([i + width for i in x], bert_scores, width, label='BERTScore F1', color='#f39c12')
        
        plt.xlabel('Question Type', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Performance by Question Type', fontsize=14, fontweight='bold')
        plt.xticks(x, types, rotation=15)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_by_type.png", dpi=300)
        plt.close()
    
    def _plot_performance_by_difficulty(self, output_dir: str):
        """Plot performance by difficulty level"""
        by_diff = self.aggregated_results['by_difficulty']
        
        if not by_diff:
            return
        
        difficulties = list(by_diff.keys())
        mrr_scores = [by_diff[d]['mrr'] for d in difficulties]
        ndcg_scores = [by_diff[d]['ndcg_5'] for d in difficulties]
        
        plt.figure(figsize=(10, 6))
        x = range(len(difficulties))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], mrr_scores, width, label='MRR', color='#e74c3c')
        plt.bar([i + width/2 for i in x], ndcg_scores, width, label='NDCG@5', color='#9b59b6')
        
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Performance by Difficulty', fontsize=14, fontweight='bold')
        plt.xticks(x, difficulties)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_by_difficulty.png", dpi=300)
        plt.close()
    
    def _plot_time_distribution(self, output_dir: str):
        """Plot response time distribution"""
        retrieval_times = [r['retrieval_time'] for r in self.evaluation_results]
        generation_times = [r['generation_time'] for r in self.evaluation_results]
        total_times = [r['total_time'] for r in self.evaluation_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(retrieval_times, bins=20, color='#3498db', edgecolor='black')
        axes[0].set_title('Retrieval Time Distribution')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(generation_times, bins=20, color='#2ecc71', edgecolor='black')
        axes[1].set_title('Generation Time Distribution')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Frequency')
        
        axes[2].hist(total_times, bins=20, color='#f39c12', edgecolor='black')
        axes[2].set_title('Total Time Distribution')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_distribution.png", dpi=300)
        plt.close()
    
    def _plot_score_distributions(self, output_dir: str):
        """Plot score distributions"""
        mrr_scores = [r['metrics']['mrr'] for r in self.evaluation_results]
        ndcg_scores = [r['metrics']['ndcg_5'] for r in self.evaluation_results]
        bert_scores = [r['metrics']['bertscore_f1'] for r in self.evaluation_results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(mrr_scores, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(mrr_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mrr_scores):.3f}')
        axes[0].set_title('MRR Score Distribution')
        axes[0].set_xlabel('MRR Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        axes[1].hist(ndcg_scores, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(ndcg_scores), color='purple', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ndcg_scores):.3f}')
        axes[1].set_title('NDCG@5 Score Distribution')
        axes[1].set_xlabel('NDCG@5 Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        axes[2].hist(bert_scores, bins=20, color='#16a085', edgecolor='black', alpha=0.7)
        axes[2].axvline(np.mean(bert_scores), color='teal', linestyle='--', linewidth=2, label=f'Mean: {np.mean(bert_scores):.3f}')
        axes[2].set_title('BERTScore F1 Distribution')
        axes[2].set_xlabel('BERTScore F1')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_distributions.png", dpi=300)
        plt.close()
    
    def save_results(self, output_dir: str = "reports"):
        """Save all results to files"""
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Save detailed results JSON
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'overall_metrics': self.aggregated_results['overall_metrics'],
            'by_question_type': self.aggregated_results['by_question_type'],
            'by_difficulty': self.aggregated_results['by_difficulty'],
            'time_statistics': self.aggregated_results['time_statistics'],
            'detailed_results': self.evaluation_results
        }
        
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Saved detailed results to {output_dir}/evaluation_results.json")
        
        # 2. Save CSV for easy analysis
        df_data = []
        for result in self.evaluation_results:
            row = {
                'Question_ID': result['question_id'],
                'Question': result['question'],
                'Question_Type': result['question_type'],
                'Difficulty': result['difficulty'],
                'MRR': result['metrics']['mrr'],
                'NDCG@5': result['metrics']['ndcg_5'],
                'BERTScore_F1': result['metrics']['bertscore_f1'],
                'Precision@5': result['metrics']['precision_5'],
                'Recall@5': result['metrics']['recall_5'],
                'ROUGE-L': result['metrics']['rougeL'],
                'Retrieval_Time': result['retrieval_time'],
                'Generation_Time': result['generation_time'],
                'Total_Time': result['total_time'],
                'Generated_Answer': result['generated_answer'][:100] + "..."  # Truncate
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
        
        print(f"✓ Saved CSV to {output_dir}/evaluation_results.csv")
        
        # 3. Save metric explanations
        explanations = self.evaluator.generate_metric_explanations()
        with open(f"{output_dir}/metric_explanations.txt", 'w') as f:
            for metric_name, explanation in explanations.items():
                f.write(f"\n{'='*80}\n")
                f.write(explanation)
                f.write(f"\n{'='*80}\n\n")
        
        print(f"✓ Saved metric explanations to {output_dir}/metric_explanations.txt")
    
    def run_full_pipeline(self):
        """Run the complete evaluation pipeline"""
        print("\n" + "="*80)
        print("HYBRID RAG SYSTEM - AUTOMATED EVALUATION PIPELINE")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 1. Load questions
            questions = self.load_questions()
            
            # 2. Initialize RAG system
            self.initialize_rag_system()
            
            # 3. Run main evaluation
            self.run_evaluation(questions, method="hybrid")
            
            # 4. Run ablation study (on subset)
            self.run_ablation_study(questions[:20])
            
            # 5. Run error analysis
            self.run_error_analysis()
            
            # 6. Generate visualizations
            self.generate_visualizations()
            
            # 7. Save all results
            self.save_results()
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("EVALUATION PIPELINE COMPLETE")
            print(f"{'='*80}")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"\nResults saved to reports/ directory")
            print(f"  - evaluation_results.json (detailed results)")
            print(f"  - evaluation_results.csv (tabular format)")
            print(f"  - visualizations/ (plots and charts)")
            print(f"  - ablation/ (ablation study results)")
            print(f"  - errors/ (error analysis)")
            print(f"{'='*80}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error in evaluation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    pipeline = EvaluationPipeline()
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n✓ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
