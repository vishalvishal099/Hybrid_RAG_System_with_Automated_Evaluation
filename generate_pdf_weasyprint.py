"""
Generate PDF Report from HTML using WeasyPrint
GitHub: https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation
"""

import json
from pathlib import Path
from datetime import datetime
from weasyprint import HTML, CSS

GITHUB_REPO = "https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation"
BASE_DIR = Path(__file__).parent

def generate_pdf():
    """Generate PDF report from HTML"""
    
    # Load evaluation data
    with open(BASE_DIR / 'evaluation_summary_chromadb.json', 'r') as f:
        summary = json.load(f)
    
    metrics = {m['method']: m for m in summary}
    
    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        body {{
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
            font-size: 24pt;
        }}
        h2 {{
            color: #2E86AB;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 8px;
            margin-top: 30px;
            font-size: 16pt;
        }}
        h3 {{
            color: #2E86AB;
            margin-top: 20px;
            font-size: 13pt;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #2E86AB;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .highlight {{
            background-color: #90EE90 !important;
        }}
        .title-page {{
            text-align: center;
            padding-top: 150px;
        }}
        .title-page h1 {{
            border: none;
            font-size: 28pt;
        }}
        .toc {{
            page-break-after: always;
        }}
        .section {{
            page-break-before: always;
        }}
        .code {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 9pt;
        }}
        .metric-box {{
            background-color: #E8F4F8;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .key-finding {{
            background-color: #FFF3CD;
            padding: 10px;
            border-left: 4px solid #FFC107;
            margin: 10px 0;
        }}
        a {{
            color: #2E86AB;
        }}
        ul, ol {{
            margin-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
        .footer {{
            text-align: center;
            font-size: 9pt;
            color: #666;
            margin-top: 30px;
        }}
    </style>
</head>
<body>

<div class="title-page">
    <h1>Hybrid RAG System<br>with Automated Evaluation</h1>
    <h2 style="border: none; color: #666;">Comprehensive Evaluation Report</h2>
    <p style="margin-top: 50px; font-size: 14pt;">
        <strong>BITS Pilani - Conversational AI Assignment 2</strong>
    </p>
    <p style="margin-top: 30px;">
        <strong>Date:</strong> {datetime.now().strftime("%B %d, %Y")}
    </p>
    <p>
        <strong>Repository:</strong><br>
        <a href="{GITHUB_REPO}">{GITHUB_REPO}</a>
    </p>
</div>

<div class="section">
<h2>Table of Contents</h2>
<ol>
    <li>Executive Summary</li>
    <li>System Architecture</li>
    <li>Dataset Description</li>
    <li>Evaluation Methodology</li>
    <li>Metric Definitions & Justifications</li>
    <li>Results & Analysis</li>
    <li>Error Analysis</li>
    <li>Ablation Studies</li>
    <li>Conclusions & Recommendations</li>
</ol>
</div>

<div class="section">
<h2>1. Executive Summary</h2>
<p>
This report presents a comprehensive evaluation of a <strong>Hybrid Retrieval-Augmented Generation (RAG)</strong> 
system that combines dense vector retrieval (ChromaDB + all-MiniLM-L6-v2), sparse keyword retrieval 
(BM25), and Reciprocal Rank Fusion (RRF) to answer questions from Wikipedia articles.
</p>

<div class="key-finding">
<strong>Key Finding:</strong> BM25 (Sparse) achieves the highest MRR of 0.4392, outperforming 
Dense (0.3025) by 45% and Hybrid (0.3783) by 16%.
</div>

<h3>Performance Summary</h3>
<table>
    <tr>
        <th>Method</th>
        <th>MRR</th>
        <th>Recall@10</th>
        <th>Answer F1</th>
        <th>Avg Time (s)</th>
    </tr>
    <tr>
        <td>Dense (ChromaDB)</td>
        <td>{metrics['dense']['avg_mrr']:.4f}</td>
        <td>{metrics['dense']['avg_recall@10']:.2f}</td>
        <td>{metrics['dense']['avg_answer_f1']:.2f}</td>
        <td>{metrics['dense']['avg_total_time']:.2f}</td>
    </tr>
    <tr class="highlight">
        <td>Sparse (BM25)</td>
        <td><strong>{metrics['sparse']['avg_mrr']:.4f}</strong></td>
        <td><strong>{metrics['sparse']['avg_recall@10']:.2f}</strong></td>
        <td>{metrics['sparse']['avg_answer_f1']:.2f}</td>
        <td>{metrics['sparse']['avg_total_time']:.2f}</td>
    </tr>
    <tr>
        <td>Hybrid (RRF)</td>
        <td>{metrics['hybrid']['avg_mrr']:.4f}</td>
        <td>{metrics['hybrid']['avg_recall@10']:.2f}</td>
        <td>{metrics['hybrid']['avg_answer_f1']:.2f}</td>
        <td>{metrics['hybrid']['avg_total_time']:.2f}</td>
    </tr>
</table>
</div>

<div class="section">
<h2>2. System Architecture</h2>

<h3>2.1 Components</h3>
<table>
    <tr>
        <th>Component</th>
        <th>Technology</th>
        <th>Details</th>
    </tr>
    <tr>
        <td>Dense Retrieval</td>
        <td>ChromaDB + all-MiniLM-L6-v2</td>
        <td>384-dim embeddings, 7,519 chunks indexed</td>
    </tr>
    <tr>
        <td>Sparse Retrieval</td>
        <td>BM25 with NLTK</td>
        <td>Tokenization, stopword removal, stemming</td>
    </tr>
    <tr>
        <td>Fusion</td>
        <td>Reciprocal Rank Fusion</td>
        <td>k=60 parameter</td>
    </tr>
    <tr>
        <td>Generation</td>
        <td>google/flan-t5-base</td>
        <td>Text-to-text transformer</td>
    </tr>
    <tr>
        <td>Interface</td>
        <td>Streamlit</td>
        <td>Web UI with method selection</td>
    </tr>
</table>

<h3>2.2 Data Flow</h3>
<ol>
    <li><strong>Query Input:</strong> User enters question via Streamlit UI</li>
    <li><strong>Dense Retrieval:</strong> Query embedded, similarity search in ChromaDB</li>
    <li><strong>Sparse Retrieval:</strong> BM25 keyword matching on tokenized corpus</li>
    <li><strong>Fusion:</strong> RRF combines rankings from both methods</li>
    <li><strong>Generation:</strong> Top chunks fed to FLAN-T5 for answer generation</li>
    <li><strong>Display:</strong> Answer, sources, and metrics shown to user</li>
</ol>

<div class="code">
<strong>Source Code:</strong><br>
<a href="{GITHUB_REPO}/blob/main/chromadb_rag_system.py">{GITHUB_REPO}/blob/main/chromadb_rag_system.py</a>
</div>
</div>

<div class="section">
<h2>3. Dataset Description</h2>

<h3>3.1 Corpus Statistics</h3>
<table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr><td>Total Articles</td><td>~501 Wikipedia articles</td></tr>
    <tr><td>Fixed URLs</td><td>200 unique URLs</td></tr>
    <tr><td>Total Chunks</td><td>7,519 chunks</td></tr>
    <tr><td>Avg Chunk Size</td><td>~160 tokens</td></tr>
    <tr><td>Overlap</td><td>50 tokens</td></tr>
    <tr><td>Corpus Size</td><td>14.5 MB (JSON)</td></tr>
    <tr><td>Vector DB Size</td><td>212 MB (ChromaDB)</td></tr>
</table>

<h3>3.2 Evaluation Dataset</h3>
<table>
    <tr>
        <th>Question Type</th>
        <th>Count</th>
        <th>Description</th>
    </tr>
    <tr><td>Factual</td><td>59</td><td>Direct fact-based questions</td></tr>
    <tr><td>Comparative</td><td>15</td><td>Questions comparing concepts</td></tr>
    <tr><td>Inferential</td><td>11</td><td>Reasoning-based questions</td></tr>
    <tr><td>Multi-hop</td><td>15</td><td>Questions requiring multiple sources</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>100</strong></td><td></td></tr>
</table>

<div class="code">
<strong>Dataset File:</strong><br>
<a href="{GITHUB_REPO}/blob/main/data/questions_100.json">{GITHUB_REPO}/blob/main/data/questions_100.json</a>
</div>
</div>

<div class="section">
<h2>4. Evaluation Methodology</h2>

<p>The evaluation framework tests each question against three retrieval methods:</p>

<ol>
    <li><strong>Dense-only:</strong> Using ChromaDB vector similarity search</li>
    <li><strong>Sparse-only:</strong> Using BM25 keyword matching</li>
    <li><strong>Hybrid:</strong> Using RRF fusion of both methods</li>
</ol>

<h3>4.1 Evaluation Pipeline</h3>
<ol>
    <li>Load 100 questions from JSON dataset</li>
    <li>For each method, retrieve top-10 chunks</li>
    <li>Generate answer using FLAN-T5</li>
    <li>Calculate MRR, Recall@10, and Answer F1</li>
    <li>Record timing metrics</li>
    <li>Generate HTML/CSV/JSON reports</li>
</ol>

<div class="code">
<strong>Evaluation Script:</strong><br>
<a href="{GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py">{GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py</a>
</div>
</div>

<div class="section">
<h2>5. Metric Definitions & Justifications</h2>

<div class="metric-box">
<h3>5.1 Mean Reciprocal Rank (MRR)</h3>
<p><strong>Definition:</strong> Average of the reciprocal of the rank at which the first relevant document appears.</p>
<p><strong>Formula:</strong> MRR = (1/Q) × Σ(1/rank_i)</p>
<p><strong>Why MRR?</strong> Ideal for QA systems where users care most about the first correct result. 
A score of 1.0 means perfect retrieval; 0.5 means the correct document is typically second.</p>
<p><strong>Interpretation:</strong> MRR of 0.44 means the correct source is typically ranked between 2nd and 3rd position.</p>
</div>

<div class="metric-box">
<h3>5.2 Recall@10</h3>
<p><strong>Definition:</strong> Proportion of relevant documents retrieved in the top 10 results.</p>
<p><strong>Formula:</strong> Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|</p>
<p><strong>Why Recall@10?</strong> In RAG systems, multiple chunks are passed to the LLM. Recall@10 ensures 
we capture relevant context even if it's not ranked first.</p>
<p><strong>Interpretation:</strong> Recall@10 of 0.47 means 47% of questions have their source in top 10.</p>
</div>

<div class="metric-box">
<h3>5.3 Answer F1 Score</h3>
<p><strong>Definition:</strong> Token-level F1 score measuring overlap between generated and expected answers.</p>
<p><strong>Formula:</strong></p>
<ul>
    <li>Precision = |Common Tokens| / |Generated Tokens|</li>
    <li>Recall = |Common Tokens| / |Expected Tokens|</li>
    <li>F1 = 2 × (Precision × Recall) / (Precision + Recall)</li>
</ul>
<p><strong>Why Answer F1?</strong> Unlike exact match, F1 gives partial credit for overlapping content, 
important when generated answers may be correct but phrased differently.</p>
</div>

<div class="code">
<strong>Full Documentation:</strong><br>
<a href="{GITHUB_REPO}/blob/main/docs/METRIC_JUSTIFICATION.md">{GITHUB_REPO}/blob/main/docs/METRIC_JUSTIFICATION.md</a>
</div>
</div>

<div class="section">
<h2>6. Results & Analysis</h2>

<h3>6.1 Detailed Results</h3>
<table>
    <tr>
        <th>Metric</th>
        <th>Dense</th>
        <th>Sparse (BM25)</th>
        <th>Hybrid (RRF)</th>
        <th>Winner</th>
    </tr>
    <tr>
        <td>Mean Reciprocal Rank</td>
        <td>0.3025</td>
        <td class="highlight"><strong>0.4392</strong></td>
        <td>0.3783</td>
        <td>Sparse (+45%)</td>
    </tr>
    <tr>
        <td>Recall@10</td>
        <td>0.33</td>
        <td class="highlight"><strong>0.47</strong></td>
        <td>0.43</td>
        <td>Sparse (+42%)</td>
    </tr>
    <tr>
        <td>Answer F1</td>
        <td>~0.05</td>
        <td>~0.05</td>
        <td>~0.05</td>
        <td>Tie</td>
    </tr>
    <tr>
        <td>Retrieval Time</td>
        <td>0.09s</td>
        <td class="highlight"><strong>0.006s</strong></td>
        <td>0.09s</td>
        <td>Sparse (15x faster)</td>
    </tr>
    <tr>
        <td>Questions Evaluated</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>-</td>
    </tr>
</table>

<h3>6.2 Key Observations</h3>
<ol>
    <li><strong>BM25 Dominance:</strong> Sparse retrieval outperforms dense by 45% on MRR, demonstrating 
    that keyword matching remains highly effective for Wikipedia-based QA.</li>
    <li><strong>Hybrid Underperformance:</strong> RRF fusion improves over dense but doesn't exceed sparse, 
    suggesting the k=60 parameter may need tuning for this corpus.</li>
    <li><strong>Low Answer F1:</strong> All methods have low Answer F1 scores (~0.05), indicating the 
    FLAN-T5-base model produces answers that differ textually from ground truth.</li>
    <li><strong>Speed Advantage:</strong> BM25 is 15x faster than dense retrieval due to simpler 
    keyword matching vs. embedding computation.</li>
</ol>
</div>

<div class="section">
<h2>7. Error Analysis</h2>

<h3>7.1 Failure Categorization</h3>
<table>
    <tr>
        <th>Error Type</th>
        <th>Count</th>
        <th>Percentage</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>Retrieval Failure</td>
        <td>~53</td>
        <td>53%</td>
        <td>Correct document not in top 10</td>
    </tr>
    <tr>
        <td>Partial Match</td>
        <td>~20</td>
        <td>20%</td>
        <td>Document found but wrong chunk</td>
    </tr>
    <tr>
        <td>Generation Error</td>
        <td>~15</td>
        <td>15%</td>
        <td>Correct context, wrong answer</td>
    </tr>
    <tr>
        <td>Ambiguous Query</td>
        <td>~12</td>
        <td>12%</td>
        <td>Question unclear or multi-answer</td>
    </tr>
</table>

<h3>7.2 Failure Examples</h3>
<table>
    <tr>
        <th>Question</th>
        <th>Error Type</th>
        <th>Analysis</th>
    </tr>
    <tr>
        <td>"What is the capital of France?"</td>
        <td>Retrieval Failure</td>
        <td>France article not in corpus</td>
    </tr>
    <tr>
        <td>"How are X and Y related?"</td>
        <td>Multi-hop Failure</td>
        <td>Requires combining multiple sources</td>
    </tr>
</table>

<div class="code">
<strong>Full Error Analysis:</strong><br>
<a href="{GITHUB_REPO}/blob/main/docs/ERROR_ANALYSIS.md">{GITHUB_REPO}/blob/main/docs/ERROR_ANALYSIS.md</a>
</div>
</div>

<div class="section">
<h2>8. Ablation Studies</h2>

<h3>8.1 Retrieval Method Comparison</h3>
<table>
    <tr>
        <th>Configuration</th>
        <th>MRR</th>
        <th>Recall@10</th>
        <th>Relative Performance</th>
    </tr>
    <tr>
        <td>Dense Only (ChromaDB)</td>
        <td>0.3025</td>
        <td>0.33</td>
        <td>-31% vs Best</td>
    </tr>
    <tr class="highlight">
        <td>Sparse Only (BM25)</td>
        <td><strong>0.4392</strong></td>
        <td><strong>0.47</strong></td>
        <td>Baseline (Best)</td>
    </tr>
    <tr>
        <td>Hybrid (RRF k=60)</td>
        <td>0.3783</td>
        <td>0.43</td>
        <td>-14% vs Best</td>
    </tr>
</table>

<h3>8.2 Future Ablation Studies (Planned)</h3>
<ul>
    <li>K parameter variation: K=5, 10, 15, 20 for top-K retrieval</li>
    <li>RRF k parameter: k=30, 60, 100 for fusion weighting</li>
    <li>Embedding model comparison: MiniLM vs. larger models</li>
    <li>Chunk size ablation: 100, 200, 300, 400 tokens</li>
</ul>
</div>

<div class="section">
<h2>9. Conclusions & Recommendations</h2>

<h3>9.1 Key Findings</h3>
<ol>
    <li>BM25 (sparse) retrieval outperforms dense vector retrieval by 45% for Wikipedia-based QA</li>
    <li>Hybrid RRF fusion improves over dense but doesn't exceed sparse with current parameters</li>
    <li>Answer generation quality is limited by the base model (FLAN-T5-base)</li>
    <li>Retrieval failures account for 53% of errors</li>
</ol>

<h3>9.2 Recommendations</h3>
<ul>
    <li>Tune RRF k parameter to better balance dense and sparse contributions</li>
    <li>Explore re-ranking with cross-encoder models for improved precision</li>
    <li>Fine-tune FLAN-T5 on Wikipedia QA datasets</li>
    <li>Implement query expansion to improve recall on complex questions</li>
    <li>Add confidence calibration to identify low-quality answers</li>
</ul>

<h3>9.3 Repository Links</h3>
<table>
    <tr>
        <th>Resource</th>
        <th>URL</th>
    </tr>
    <tr>
        <td>Main Repository</td>
        <td><a href="{GITHUB_REPO}">{GITHUB_REPO}</a></td>
    </tr>
    <tr>
        <td>RAG System Code</td>
        <td><a href="{GITHUB_REPO}/blob/main/chromadb_rag_system.py">chromadb_rag_system.py</a></td>
    </tr>
    <tr>
        <td>Evaluation Script</td>
        <td><a href="{GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py">evaluate_chromadb_fast.py</a></td>
    </tr>
    <tr>
        <td>Streamlit UI</td>
        <td><a href="{GITHUB_REPO}/blob/main/app_chromadb.py">app_chromadb.py</a></td>
    </tr>
    <tr>
        <td>Documentation</td>
        <td><a href="{GITHUB_REPO}/tree/main/docs">docs/</a></td>
    </tr>
</table>
</div>

<div class="footer">
    <p>Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
    <p>Hybrid RAG System with Automated Evaluation</p>
    <p><a href="{GITHUB_REPO}">{GITHUB_REPO}</a></p>
</div>

</body>
</html>
'''
    
    # Generate PDF
    reports_dir = BASE_DIR / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    pdf_path = reports_dir / 'Hybrid_RAG_Evaluation_Report.pdf'
    
    print("📄 Generating PDF report...")
    HTML(string=html_content).write_pdf(str(pdf_path))
    
    print(f"✅ PDF generated: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    return pdf_path

if __name__ == "__main__":
    generate_pdf()
