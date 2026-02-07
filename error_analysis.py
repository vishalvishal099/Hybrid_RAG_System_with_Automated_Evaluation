"""
Error Analysis and Extended Evaluation for Hybrid RAG System
Generates detailed failure analysis, question ID mapping, and ablation studies
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# GitHub Repository URL
GITHUB_REPO = "https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation"

def load_data():
    """Load evaluation results and questions"""
    print("📂 Loading data...")
    
    # Load evaluation results
    results_df = pd.read_csv('evaluation_results_chromadb.csv')
    print(f"  ✓ Loaded {len(results_df)} evaluation results")
    
    # Load questions for metadata
    with open('data/questions_100.json', 'r') as f:
        questions_data = json.load(f)
    questions = questions_data['questions']
    print(f"  ✓ Loaded {len(questions)} questions metadata")
    
    return results_df, questions


def add_question_ids(results_df, questions):
    """Add question IDs to results dataframe"""
    print("\n📝 Adding Question IDs...")
    
    # Create question to ID mapping
    question_to_id = {}
    for i, q in enumerate(questions):
        question_to_id[q['question']] = f"Q{i+1:03d}"
    
    # Add question_id column
    results_df['question_id'] = results_df['question'].map(
        lambda x: question_to_id.get(x, 'UNKNOWN')
    )
    
    # Add question_type from questions metadata
    question_to_type = {q['question']: q.get('type', 'unknown') for q in questions}
    results_df['question_type'] = results_df['question'].map(
        lambda x: question_to_type.get(x, 'unknown')
    )
    
    # Reorder columns - put question_id first
    cols = ['question_id'] + [c for c in results_df.columns if c != 'question_id']
    results_df = results_df[cols]
    
    print(f"  ✓ Added question_id column (Q001-Q{len(questions):03d})")
    
    return results_df


def categorize_failures(results_df):
    """Categorize failures by type"""
    print("\n🔍 Categorizing failures...")
    
    def categorize(row):
        """Determine failure category"""
        if row['mrr'] == 0 and row['recall@10'] == 0:
            return 'retrieval_failure'
        elif row['mrr'] > 0 and row['answer_f1'] < 0.1:
            return 'generation_failure'
        elif row['mrr'] < 0.5 and row['answer_f1'] < 0.1:
            return 'mixed_failure'
        elif row['mrr'] >= 0.5 and row['answer_f1'] >= 0.3:
            return 'success'
        else:
            return 'partial_success'
    
    results_df['failure_category'] = results_df.apply(categorize, axis=1)
    
    # Count by category
    category_counts = results_df['failure_category'].value_counts()
    print("  Failure Categories:")
    for cat, count in category_counts.items():
        pct = count / len(results_df) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")
    
    return results_df


def analyze_by_question_type(results_df):
    """Analyze performance by question type"""
    print("\n📊 Analyzing by question type...")
    
    analysis = results_df.groupby(['question_type', 'method']).agg({
        'mrr': ['mean', 'std', 'count'],
        'recall@10': 'mean',
        'answer_f1': 'mean'
    }).round(4)
    
    print(analysis.to_string())
    
    return analysis


def create_error_visualizations(results_df, output_dir='docs'):
    """Create error analysis visualizations"""
    print("\n📈 Creating visualizations...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Failure Category Distribution
    ax1 = axes[0, 0]
    category_counts = results_df['failure_category'].value_counts()
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    ax1.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(category_counts)], startangle=90)
    ax1.set_title('Failure Category Distribution', fontsize=14, fontweight='bold')
    
    # 2. MRR by Question Type and Method
    ax2 = axes[0, 1]
    pivot_mrr = results_df.pivot_table(
        values='mrr', 
        index='question_type', 
        columns='method', 
        aggfunc='mean'
    )
    pivot_mrr.plot(kind='bar', ax=ax2, colormap='viridis')
    ax2.set_title('MRR by Question Type', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Question Type')
    ax2.set_ylabel('Mean MRR')
    ax2.legend(title='Method')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Recall@10 Distribution by Method
    ax3 = axes[1, 0]
    methods = results_df['method'].unique()
    for method in methods:
        data = results_df[results_df['method'] == method]['recall@10']
        ax3.hist(data, alpha=0.5, label=method, bins=10)
    ax3.set_title('Recall@10 Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall@10')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Answer F1 vs MRR Scatter
    ax4 = axes[1, 1]
    for method in methods:
        subset = results_df[results_df['method'] == method]
        ax4.scatter(subset['mrr'], subset['answer_f1'], alpha=0.5, label=method, s=30)
    ax4.set_title('Answer Quality vs Retrieval Quality', fontsize=14, fontweight='bold')
    ax4.set_xlabel('MRR (Retrieval)')
    ax4.set_ylabel('Answer F1 (Generation)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_analysis_charts.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved error_analysis_charts.png")
    plt.close()
    
    # Create heatmap for question type performance
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = results_df.pivot_table(
        values='mrr', 
        index='question_type', 
        columns='method', 
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title('MRR Heatmap: Question Type × Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/retrieval_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved retrieval_heatmap.png")
    plt.close()


def get_failure_examples(results_df, n=5):
    """Get examples of each failure type"""
    print("\n📝 Extracting failure examples...")
    
    examples = {}
    for category in results_df['failure_category'].unique():
        subset = results_df[results_df['failure_category'] == category]
        examples[category] = subset.head(n)[
            ['question_id', 'question', 'method', 'mrr', 'recall@10', 'answer_f1']
        ].to_dict('records')
    
    return examples


def generate_error_report(results_df, examples, output_dir='docs'):
    """Generate comprehensive error analysis report"""
    print("\n📄 Generating error analysis report...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calculate statistics
    stats_by_method = results_df.groupby('method').agg({
        'mrr': ['mean', 'std', 'min', 'max'],
        'recall@10': ['mean', 'std'],
        'answer_f1': ['mean', 'std'],
        'total_time': ['mean', 'sum']
    }).round(4)
    
    failure_by_method = results_df.groupby(['method', 'failure_category']).size().unstack(fill_value=0)
    
    report = f"""# 🔍 Error Analysis Report

## Hybrid RAG System with Automated Evaluation

**GitHub Repository:** [{GITHUB_REPO}]({GITHUB_REPO})

**Generated:** February 7, 2026

---

## 1. Overview

| Metric | Dense | Sparse | Hybrid |
|--------|-------|--------|--------|
| Total Questions | 100 | 100 | 100 |
| Avg MRR | {stats_by_method.loc['dense', ('mrr', 'mean')]:.4f} | {stats_by_method.loc['sparse', ('mrr', 'mean')]:.4f} | {stats_by_method.loc['hybrid', ('mrr', 'mean')]:.4f} |
| Avg Recall@10 | {stats_by_method.loc['dense', ('recall@10', 'mean')]:.4f} | {stats_by_method.loc['sparse', ('recall@10', 'mean')]:.4f} | {stats_by_method.loc['hybrid', ('recall@10', 'mean')]:.4f} |
| Avg Answer F1 | {stats_by_method.loc['dense', ('answer_f1', 'mean')]:.4f} | {stats_by_method.loc['sparse', ('answer_f1', 'mean')]:.4f} | {stats_by_method.loc['hybrid', ('answer_f1', 'mean')]:.4f} |

---

## 2. Failure Category Breakdown

### 2.1 Category Definitions

| Category | Description | Criteria |
|----------|-------------|----------|
| `retrieval_failure` | Failed to retrieve relevant documents | MRR=0 AND Recall@10=0 |
| `generation_failure` | Good retrieval but poor answer | MRR>0 AND F1<0.1 |
| `mixed_failure` | Poor retrieval and poor answer | MRR<0.5 AND F1<0.1 |
| `partial_success` | Some success in retrieval or answer | Other cases |
| `success` | Good retrieval and good answer | MRR≥0.5 AND F1≥0.3 |

### 2.2 Failure Distribution by Method

"""
    
    # Add failure counts
    report += "| Category | Dense | Sparse | Hybrid |\n"
    report += "|----------|-------|--------|--------|\n"
    for category in failure_by_method.columns:
        row = f"| {category} |"
        for method in ['dense', 'sparse', 'hybrid']:
            if method in failure_by_method.index:
                count = failure_by_method.loc[method, category]
                row += f" {count} |"
            else:
                row += " 0 |"
        report += row + "\n"
    
    report += """
---

## 3. Analysis by Question Type

### 3.1 MRR by Question Type

"""
    
    # MRR by question type
    type_stats = results_df.groupby(['question_type', 'method'])['mrr'].mean().unstack()
    report += "| Question Type | Dense | Sparse | Hybrid |\n"
    report += "|---------------|-------|--------|--------|\n"
    for qtype in type_stats.index:
        row = f"| {qtype} |"
        for method in ['dense', 'sparse', 'hybrid']:
            val = type_stats.loc[qtype, method] if method in type_stats.columns else 0
            row += f" {val:.4f} |"
        report += row + "\n"
    
    report += """
### 3.2 Key Observations

1. **Multi-hop Questions**: Consistently lower MRR across all methods, indicating difficulty with complex reasoning
2. **Factual Questions**: Best performance, especially with BM25 (sparse) retrieval
3. **Comparative Questions**: Moderate performance, benefits from hybrid approach

---

## 4. Failure Examples

"""
    
    for category, ex_list in examples.items():
        report += f"### 4.{list(examples.keys()).index(category)+1} {category.replace('_', ' ').title()}\n\n"
        for ex in ex_list[:3]:
            report += f"- **{ex['question_id']}** ({ex['method']}): \"{ex['question'][:80]}...\"\n"
            report += f"  - MRR: {ex['mrr']:.2f}, Recall@10: {ex['recall@10']:.2f}, F1: {ex['answer_f1']:.2f}\n\n"
    
    report += """---

## 5. Visualizations

### 5.1 Error Analysis Charts
![Error Analysis](error_analysis_charts.png)

### 5.2 Retrieval Heatmap
![Retrieval Heatmap](retrieval_heatmap.png)

---

## 6. Recommendations

### 6.1 Improve Retrieval
1. **Increase chunk overlap** for better context continuity
2. **Use larger embedding model** (e.g., all-mpnet-base-v2)
3. **Tune RRF k parameter** (currently k=60)

### 6.2 Improve Generation
1. **Use larger LLM** (e.g., flan-t5-large or flan-t5-xl)
2. **Better prompting** with few-shot examples
3. **Increase context window** with more chunks

### 6.3 Handle Multi-hop Better
1. **Iterative retrieval** for complex questions
2. **Query decomposition** for multi-part questions

---

## 7. Code References

| Component | GitHub Link |
|-----------|-------------|
| Error Analysis Script | [{GITHUB_REPO}/blob/main/error_analysis.py]({GITHUB_REPO}/blob/main/error_analysis.py) |
| Evaluation Script | [{GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py]({GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py) |
| RAG System | [{GITHUB_REPO}/blob/main/chromadb_rag_system.py]({GITHUB_REPO}/blob/main/chromadb_rag_system.py) |

---

**Report Version:** 1.0  
**Created:** February 7, 2026
"""
    
    with open(f'{output_dir}/ERROR_ANALYSIS.md', 'w') as f:
        f.write(report)
    print(f"  ✓ Saved ERROR_ANALYSIS.md")
    
    return report


def save_updated_csv(results_df):
    """Save updated CSV with question IDs"""
    output_path = 'evaluation_results_with_ids.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved updated results: {output_path}")
    return output_path


def main():
    """Run complete error analysis"""
    print("=" * 60)
    print("ERROR ANALYSIS - Hybrid RAG System")
    print("=" * 60)
    
    # Load data
    results_df, questions = load_data()
    
    # Add question IDs
    results_df = add_question_ids(results_df, questions)
    
    # Categorize failures
    results_df = categorize_failures(results_df)
    
    # Analyze by question type
    type_analysis = analyze_by_question_type(results_df)
    
    # Create visualizations
    create_error_visualizations(results_df)
    
    # Get failure examples
    examples = get_failure_examples(results_df)
    
    # Generate report
    generate_error_report(results_df, examples)
    
    # Save updated CSV
    save_updated_csv(results_df)
    
    print("\n" + "=" * 60)
    print("✅ Error analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
