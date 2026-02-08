"""
Confidence Calibration Evaluation
Assesses how well model confidence correlates with answer correctness
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, '.')
sys.path.insert(0, './submission/01_source_code')

from chromadb_rag_system import ChromaDBHybridRAG
from evaluation.metrics import RAGEvaluator


def collect_confidence_data(rag_system, questions: List[Dict], sample_size: int = 100) -> List[Dict]:
    """
    Collect confidence scores and correctness for questions
    
    Returns:
        List of {confidence, correct, question, answer, ground_truth}
    """
    import random
    random.seed(42)
    
    sampled = random.sample(questions, min(sample_size, len(questions)))
    evaluator = RAGEvaluator()
    
    results = []
    
    print(f"📊 Collecting confidence data from {len(sampled)} questions...")
    
    for i, q in enumerate(sampled, 1):
        query = q['question']
        gt_answer = q.get('answer', q.get('ground_truth_answer', ''))
        
        if not gt_answer:
            continue
        
        try:
            # Get answer with confidence
            context = rag_system.retrieve_with_rrf(query, top_k=5)
            result = rag_system.generate_answer(query, context)
            
            answer = result['answer']
            confidence = result.get('confidence', 0.0)
            
            # Calculate correctness using ROUGE-L
            rouge_scores = evaluator.rouge_scorer.score(gt_answer, answer)
            correctness = rouge_scores['rougeL'].fmeasure
            
            results.append({
                'question': query,
                'answer': answer,
                'ground_truth': gt_answer,
                'confidence': confidence,
                'correctness': correctness,
                'is_correct': correctness > 0.5  # Binary threshold
            })
            
            if i % 20 == 0:
                print(f"  Processed {i}/{len(sampled)} questions...")
                
        except Exception as e:
            print(f"  ⚠ Error on question {i}: {e}")
            continue
    
    print(f"✓ Collected {len(results)} confidence-correctness pairs")
    return results


def calculate_calibration_metrics(data: List[Dict]) -> Dict:
    """
    Calculate Expected Calibration Error (ECE) and other metrics
    """
    if not data:
        return {}
    
    # Sort by confidence
    sorted_data = sorted(data, key=lambda x: x['confidence'])
    
    # Create bins (10 bins from 0-1)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # Get data in this bin
        bin_data = [d for d in data if lower <= d['confidence'] < upper or 
                    (i == n_bins - 1 and d['confidence'] == upper)]
        
        if bin_data:
            avg_confidence = np.mean([d['confidence'] for d in bin_data])
            avg_accuracy = np.mean([d['correctness'] for d in bin_data])
            count = len(bin_data)
            
            bin_confidences.append(avg_confidence)
            bin_accuracies.append(avg_accuracy)
            bin_counts.append(count)
    
    # Expected Calibration Error (ECE)
    total_samples = len(data)
    ece = sum(
        (count / total_samples) * abs(conf - acc)
        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts)
    )
    
    # Maximum Calibration Error (MCE)
    mce = max(abs(conf - acc) for conf, acc in zip(bin_confidences, bin_accuracies))
    
    # Correlation between confidence and correctness
    confidences = [d['confidence'] for d in data]
    correctnesses = [d['correctness'] for d in data]
    correlation = np.corrcoef(confidences, correctnesses)[0, 1]
    
    return {
        'ece': ece,
        'mce': mce,
        'correlation': correlation,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'n_bins': n_bins,
        'total_samples': total_samples,
        'mean_confidence': np.mean(confidences),
        'mean_correctness': np.mean(correctnesses)
    }


def plot_calibration_curve(metrics: Dict, output_path: str):
    """
    Generate reliability diagram (calibration curve)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Calibration Curve
    bin_confs = metrics['bin_confidences']
    bin_accs = metrics['bin_accuracies']
    bin_counts = metrics['bin_counts']
    
    # Plot bars
    x_pos = range(len(bin_confs))
    colors = ['green' if abs(c - a) < 0.1 else 'orange' if abs(c - a) < 0.2 else 'red'
              for c, a in zip(bin_confs, bin_accs)]
    
    ax1.bar(x_pos, bin_accs, width=0.8, alpha=0.6, color=colors, label='Actual Accuracy')
    ax1.plot(x_pos, bin_confs, 'b--', marker='o', linewidth=2, markersize=8, label='Predicted Confidence')
    
    # Perfect calibration line
    ax1.plot(x_pos, bin_confs, 'g:', linewidth=1, alpha=0.5, label='Perfect Calibration')
    
    ax1.set_xlabel('Confidence Bin', fontsize=12)
    ax1.set_ylabel('Accuracy / Confidence', fontsize=12)
    ax1.set_title(f'Calibration Curve (ECE={metrics["ece"]:.3f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add bin counts as text
    for i, count in enumerate(bin_counts):
        ax1.text(i, 0.05, f'n={count}', ha='center', fontsize=8, color='black')
    
    # Plot 2: Histogram of Confidence vs Correctness
    all_confidences = []
    all_correctnesses = []
    
    # We need raw data, but we can approximate from bins
    ax2.scatter(bin_confs, bin_accs, s=[c*10 for c in bin_counts], alpha=0.6, c=colors)
    
    # Perfect calibration diagonal
    ax2.plot([0, 1], [0, 1], 'g--', linewidth=2, label='Perfect Calibration')
    
    ax2.set_xlabel('Model Confidence', fontsize=12)
    ax2.set_ylabel('Actual Correctness (ROUGE-L F1)', fontsize=12)
    ax2.set_title(f'Confidence vs Correctness (r={metrics["correlation"]:.3f})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved calibration plot to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("CONFIDENCE CALIBRATION EVALUATION")
    print("="*70)
    
    # Initialize RAG system
    print("\n🔧 Initializing RAG system...")
    rag_system = ChromaDBHybridRAG(
        collection_name="wikipedia_articles",
        persist_directory="chroma_db"
    )
    print(f"✓ System ready with {len(rag_system.corpus)} chunks")
    
    # Load questions
    print("\n📥 Loading test questions...")
    questions_file = Path("data/questions_100.json")
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"✓ Loaded {len(questions)} questions")
    
    # Collect confidence data
    print("\n📊 Collecting confidence-correctness data...")
    data = collect_confidence_data(rag_system, questions, sample_size=100)
    
    if not data:
        print("❌ No data collected!")
        return
    
    # Calculate metrics
    print("\n📈 Calculating calibration metrics...")
    metrics = calculate_calibration_metrics(data)
    
    # Save results
    output_dir = Path("evaluation/confidence_calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'sample_data': data[:10]  # First 10 for inspection
    }
    
    results_file = output_dir / "calibration_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to: {results_file}")
    
    # Generate plots
    plot_file = output_dir / "calibration_curve.png"
    plot_calibration_curve(metrics, str(plot_file))
    
    # Display summary
    print("\n" + "="*70)
    print("📊 CALIBRATION SUMMARY")
    print("="*70)
    print(f"Total Questions: {metrics['total_samples']}")
    print(f"Number of Bins: {metrics['n_bins']}")
    print(f"\nCalibration Metrics:")
    print(f"  Expected Calibration Error (ECE): {metrics['ece']:.4f}")
    print(f"  Maximum Calibration Error (MCE): {metrics['mce']:.4f}")
    print(f"  Confidence-Correctness Correlation: {metrics['correlation']:.4f}")
    print(f"\nAverage Values:")
    print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
    print(f"  Mean Correctness: {metrics['mean_correctness']:.4f}")
    
    # Interpretation
    ece = metrics['ece']
    if ece < 0.05:
        grade = "🏆 EXCELLENT - Well calibrated"
    elif ece < 0.10:
        grade = "✅ GOOD - Reasonably calibrated"
    elif ece < 0.15:
        grade = "⚠️  MODERATE - Some miscalibration"
    else:
        grade = "❌ POOR - Poorly calibrated"
    
    print(f"\n  Calibration Quality: {grade}")
    print("\n💡 Interpretation:")
    print("  - ECE close to 0 = perfect calibration")
    print("  - Positive correlation = confidence increases with correctness")
    print("  - ECE < 0.10 is considered well-calibrated")
    print("="*70)
    
    print("\n✅ Confidence calibration evaluation complete!")


if __name__ == "__main__":
    main()
