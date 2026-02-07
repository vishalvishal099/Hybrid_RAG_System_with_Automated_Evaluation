"""
Generate PDF Report using ReportLab (no external dependencies)
GitHub: https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation
"""

import json
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

GITHUB_REPO = "https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation"
BASE_DIR = Path(__file__).parent

def create_styles():
    """Create custom styles for the report"""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Title'],
        fontSize=28,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=20,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='SubTitle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#666666'),
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2E86AB'),
        spaceBefore=20,
        spaceAfter=12,
        borderWidth=2,
        borderColor=colors.HexColor('#2E86AB'),
        borderPadding=5
    ))
    
    styles.add(ParagraphStyle(
        name='SubSection',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2E86AB'),
        spaceBefore=15,
        spaceAfter=8
    ))
    
    styles.add(ParagraphStyle(
        name='BodyTextCustom',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    ))
    
    styles.add(ParagraphStyle(
        name='BulletCustom',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=5
    ))
    
    styles.add(ParagraphStyle(
        name='CodeBlock',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Courier',
        backColor=colors.HexColor('#F4F4F4'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=10
    ))
    
    styles.add(ParagraphStyle(
        name='KeyFinding',
        parent=styles['Normal'],
        fontSize=10,
        backColor=colors.HexColor('#FFF3CD'),
        borderWidth=1,
        borderColor=colors.HexColor('#FFC107'),
        borderPadding=10,
        leftIndent=10,
        spaceAfter=15
    ))
    
    return styles

def create_table_style():
    """Create standard table style"""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
    ])

def generate_pdf():
    """Generate comprehensive PDF evaluation report"""
    
    # Load evaluation data
    with open(BASE_DIR / 'evaluation_summary_chromadb.json', 'r') as f:
        summary = json.load(f)
    
    metrics = {m['method']: m for m in summary}
    
    # Create PDF document
    reports_dir = BASE_DIR / 'reports'
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / 'Hybrid_RAG_Evaluation_Report.pdf'
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = create_styles()
    story = []
    
    # ==================== TITLE PAGE ====================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Hybrid RAG System", styles['MainTitle']))
    story.append(Paragraph("with Automated Evaluation", styles['MainTitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Comprehensive Evaluation Report", styles['SubTitle']))
    story.append(Spacer(1, 1*inch))
    
    # Project info
    info_data = [
        ["Project", "Hybrid RAG System with Automated Evaluation"],
        ["Course", "BITS Pilani - Conversational AI"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
        ["Questions", "100 Questions × 3 Methods = 300 Evaluations"],
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"<b>GitHub:</b> {GITHUB_REPO}", styles['BodyTextCustom']))
    story.append(PageBreak())
    
    # ==================== TABLE OF CONTENTS ====================
    story.append(Paragraph("Table of Contents", styles['SectionHeader']))
    toc = [
        "1. Executive Summary",
        "2. System Architecture", 
        "3. Dataset Description",
        "4. Evaluation Methodology",
        "5. Metric Definitions & Justifications",
        "6. Results & Analysis",
        "7. Error Analysis",
        "8. Ablation Studies",
        "9. Conclusions & Recommendations",
    ]
    for item in toc:
        story.append(Paragraph(item, styles['BodyTextCustom']))
    story.append(PageBreak())
    
    # ==================== 1. EXECUTIVE SUMMARY ====================
    story.append(Paragraph("1. Executive Summary", styles['SectionHeader']))
    
    summary_text = """
    This report presents a comprehensive evaluation of a <b>Hybrid Retrieval-Augmented Generation (RAG)</b> 
    system that combines dense vector retrieval (ChromaDB + all-MiniLM-L6-v2), sparse keyword retrieval 
    (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from Wikipedia articles.
    """
    story.append(Paragraph(summary_text, styles['BodyTextCustom']))
    
    story.append(Paragraph(
        "<b>Key Finding:</b> BM25 (Sparse) achieves the highest MRR of 0.4392, outperforming "
        "Dense (0.3025) by 45% and Hybrid (0.3783) by 16%.",
        styles['KeyFinding']
    ))
    
    # Performance summary table
    story.append(Paragraph("<b>Performance Summary:</b>", styles['SubSection']))
    perf_data = [
        ["Method", "MRR", "Recall@10", "Avg Time (s)"],
        ["Dense (ChromaDB)", f"{metrics['dense']['avg_mrr']:.4f}", f"{metrics['dense']['avg_recall@10']:.2f}", f"{metrics['dense']['avg_total_time']:.2f}"],
        ["Sparse (BM25)", f"{metrics['sparse']['avg_mrr']:.4f}", f"{metrics['sparse']['avg_recall@10']:.2f}", f"{metrics['sparse']['avg_total_time']:.2f}"],
        ["Hybrid (RRF)", f"{metrics['hybrid']['avg_mrr']:.4f}", f"{metrics['hybrid']['avg_recall@10']:.2f}", f"{metrics['hybrid']['avg_total_time']:.2f}"],
    ]
    perf_table = Table(perf_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    ts = create_table_style()
    ts.add('BACKGROUND', (1, 2), (2, 2), colors.HexColor('#90EE90'))  # Highlight best
    perf_table.setStyle(ts)
    story.append(perf_table)
    story.append(PageBreak())
    
    # ==================== 2. SYSTEM ARCHITECTURE ====================
    story.append(Paragraph("2. System Architecture", styles['SectionHeader']))
    
    story.append(Paragraph("<b>2.1 Components</b>", styles['SubSection']))
    comp_data = [
        ["Component", "Technology", "Details"],
        ["Dense Retrieval", "ChromaDB + MiniLM", "384-dim embeddings, 7,519 chunks"],
        ["Sparse Retrieval", "BM25 + NLTK", "Tokenization, stopwords, stemming"],
        ["Fusion", "RRF", "k=60 parameter"],
        ["Generation", "FLAN-T5-base", "Text-to-text transformer"],
        ["Interface", "Streamlit", "Web UI with method selection"],
    ]
    comp_table = Table(comp_data, colWidths=[1.5*inch, 1.8*inch, 2.5*inch])
    comp_table.setStyle(create_table_style())
    story.append(comp_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>2.2 Data Flow</b>", styles['SubSection']))
    flow_items = [
        "1. Query Input: User enters question via Streamlit UI",
        "2. Dense Retrieval: Query embedded, similarity search in ChromaDB",
        "3. Sparse Retrieval: BM25 keyword matching on tokenized corpus",
        "4. Fusion: RRF combines rankings from both methods",
        "5. Generation: Top chunks fed to FLAN-T5 for answer generation",
        "6. Display: Answer, sources, and metrics shown to user",
    ]
    for item in flow_items:
        story.append(Paragraph(f"• {item}", styles['BulletCustom']))
    
    story.append(Paragraph(f"<b>Source:</b> {GITHUB_REPO}/blob/main/chromadb_rag_system.py", styles['CodeBlock']))
    story.append(PageBreak())
    
    # ==================== 3. DATASET DESCRIPTION ====================
    story.append(Paragraph("3. Dataset Description", styles['SectionHeader']))
    
    story.append(Paragraph("<b>3.1 Corpus Statistics</b>", styles['SubSection']))
    corpus_data = [
        ["Metric", "Value"],
        ["Total Articles", "~501 Wikipedia articles"],
        ["Fixed URLs", "200 unique URLs"],
        ["Total Chunks", "7,519 chunks"],
        ["Avg Chunk Size", "~160 tokens"],
        ["Overlap", "50 tokens"],
        ["Corpus Size", "14.5 MB (JSON)"],
        ["Vector DB Size", "212 MB (ChromaDB)"],
    ]
    corpus_table = Table(corpus_data, colWidths=[2*inch, 3*inch])
    corpus_table.setStyle(create_table_style())
    story.append(corpus_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>3.2 Evaluation Questions</b>", styles['SubSection']))
    q_data = [
        ["Question Type", "Count", "Description"],
        ["Factual", "59", "Direct fact-based questions"],
        ["Comparative", "15", "Questions comparing concepts"],
        ["Inferential", "11", "Reasoning-based questions"],
        ["Multi-hop", "15", "Questions requiring multiple sources"],
        ["Total", "100", "-"],
    ]
    q_table = Table(q_data, colWidths=[1.5*inch, 1*inch, 3*inch])
    q_table.setStyle(create_table_style())
    story.append(q_table)
    story.append(PageBreak())
    
    # ==================== 4. EVALUATION METHODOLOGY ====================
    story.append(Paragraph("4. Evaluation Methodology", styles['SectionHeader']))
    
    method_text = """
    The evaluation framework tests each of the 100 questions against three retrieval methods:
    Dense-only (ChromaDB), Sparse-only (BM25), and Hybrid (RRF fusion). For each query:
    """
    story.append(Paragraph(method_text, styles['BodyTextCustom']))
    
    steps = [
        "1. Retrieve top-10 chunks using the selected method",
        "2. Generate answer using FLAN-T5 with retrieved context",
        "3. Calculate MRR based on ground truth URL ranking",
        "4. Calculate Recall@10 for source retrieval",
        "5. Calculate Answer F1 for generation quality",
        "6. Record timing metrics for performance analysis",
    ]
    for step in steps:
        story.append(Paragraph(f"• {step}", styles['BulletCustom']))
    
    story.append(Paragraph(f"<b>Script:</b> {GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py", styles['CodeBlock']))
    story.append(PageBreak())
    
    # ==================== 5. METRIC DEFINITIONS ====================
    story.append(Paragraph("5. Metric Definitions & Justifications", styles['SectionHeader']))
    
    # MRR
    story.append(Paragraph("<b>5.1 Mean Reciprocal Rank (MRR)</b>", styles['SubSection']))
    story.append(Paragraph(
        "<b>Definition:</b> Average of the reciprocal of the rank at which the first relevant document appears.",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        "<b>Formula:</b> MRR = (1/Q) × Σ(1/rank_i) where Q is number of queries",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        "<b>Why MRR?</b> Ideal for QA systems where users care most about the first correct result. "
        "A score of 1.0 means perfect retrieval; 0.5 means the correct document is typically second.",
        styles['BodyTextCustom']
    ))
    
    # Recall@10
    story.append(Paragraph("<b>5.2 Recall@10</b>", styles['SubSection']))
    story.append(Paragraph(
        "<b>Definition:</b> Proportion of relevant documents retrieved in the top 10 results.",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        "<b>Formula:</b> Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        "<b>Why Recall@10?</b> In RAG systems, multiple chunks are passed to the LLM. "
        "Recall@10 ensures we capture relevant context even if not ranked first.",
        styles['BodyTextCustom']
    ))
    
    # Answer F1
    story.append(Paragraph("<b>5.3 Answer F1 Score</b>", styles['SubSection']))
    story.append(Paragraph(
        "<b>Definition:</b> Token-level F1 score measuring overlap between generated and expected answers.",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        "<b>Why Answer F1?</b> Unlike exact match, F1 gives partial credit for overlapping content, "
        "important when generated answers may be correct but phrased differently.",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(f"<b>Full Documentation:</b> {GITHUB_REPO}/blob/main/docs/METRIC_JUSTIFICATION.md", styles['CodeBlock']))
    story.append(PageBreak())
    
    # ==================== 6. RESULTS & ANALYSIS ====================
    story.append(Paragraph("6. Results & Analysis", styles['SectionHeader']))
    
    story.append(Paragraph("<b>6.1 Detailed Results</b>", styles['SubSection']))
    results_data = [
        ["Metric", "Dense", "Sparse", "Hybrid", "Best"],
        ["MRR", "0.3025", "0.4392", "0.3783", "Sparse (+45%)"],
        ["Recall@10", "0.33", "0.47", "0.43", "Sparse (+42%)"],
        ["Answer F1", "~0.05", "~0.05", "~0.05", "Tie"],
        ["Retrieval Time", "0.09s", "0.006s", "0.09s", "Sparse (15x)"],
    ]
    results_table = Table(results_data, colWidths=[1.3*inch, 1*inch, 1*inch, 1*inch, 1.3*inch])
    ts = create_table_style()
    ts.add('BACKGROUND', (2, 1), (2, 2), colors.HexColor('#90EE90'))
    results_table.setStyle(ts)
    story.append(results_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>6.2 Key Observations</b>", styles['SubSection']))
    observations = [
        "BM25 Dominance: Sparse retrieval outperforms dense by 45% on MRR",
        "Hybrid Underperformance: RRF doesn't exceed sparse with current k=60",
        "Low Answer F1: All methods have ~0.05 F1, suggesting model limitations",
        "Speed: BM25 is 15x faster than dense retrieval",
    ]
    for obs in observations:
        story.append(Paragraph(f"• {obs}", styles['BulletCustom']))
    story.append(PageBreak())
    
    # ==================== 7. ERROR ANALYSIS ====================
    story.append(Paragraph("7. Error Analysis", styles['SectionHeader']))
    
    error_data = [
        ["Error Type", "Count", "%", "Description"],
        ["Retrieval Failure", "~53", "53%", "Correct doc not in top 10"],
        ["Partial Match", "~20", "20%", "Doc found, wrong chunk"],
        ["Generation Error", "~15", "15%", "Correct context, wrong answer"],
        ["Ambiguous Query", "~12", "12%", "Question unclear"],
    ]
    error_table = Table(error_data, colWidths=[1.5*inch, 0.8*inch, 0.6*inch, 2.5*inch])
    error_table.setStyle(create_table_style())
    story.append(error_table)
    story.append(Paragraph(f"<b>Details:</b> {GITHUB_REPO}/blob/main/docs/ERROR_ANALYSIS.md", styles['CodeBlock']))
    story.append(PageBreak())
    
    # ==================== 8. ABLATION STUDIES ====================
    story.append(Paragraph("8. Ablation Studies", styles['SectionHeader']))
    
    ablation_data = [
        ["Configuration", "MRR", "Recall@10", "vs Best"],
        ["Dense Only", "0.3025", "0.33", "-31%"],
        ["Sparse Only (Best)", "0.4392", "0.47", "Baseline"],
        ["Hybrid (RRF k=60)", "0.3783", "0.43", "-14%"],
    ]
    ablation_table = Table(ablation_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    ts = create_table_style()
    ts.add('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#90EE90'))
    ablation_table.setStyle(ts)
    story.append(ablation_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Future Ablation Studies:</b>", styles['SubSection']))
    future = [
        "K parameter: K=5, 10, 15, 20 for top-K retrieval",
        "RRF k: k=30, 60, 100 for fusion weighting",
        "Embedding models: MiniLM vs larger models",
        "Chunk sizes: 100, 200, 300, 400 tokens",
    ]
    for f in future:
        story.append(Paragraph(f"• {f}", styles['BulletCustom']))
    story.append(PageBreak())
    
    # ==================== 9. CONCLUSIONS ====================
    story.append(Paragraph("9. Conclusions & Recommendations", styles['SectionHeader']))
    
    story.append(Paragraph("<b>9.1 Key Findings</b>", styles['SubSection']))
    findings = [
        "BM25 (sparse) outperforms dense vector retrieval by 45% for Wikipedia QA",
        "Hybrid RRF doesn't exceed sparse performance with current parameters",
        "Answer generation limited by FLAN-T5-base model",
        "Retrieval failures account for 53% of errors",
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", styles['BulletCustom']))
    
    story.append(Paragraph("<b>9.2 Recommendations</b>", styles['SubSection']))
    recs = [
        "Tune RRF k parameter for better fusion",
        "Add cross-encoder re-ranking for precision",
        "Fine-tune FLAN-T5 on Wikipedia QA",
        "Implement query expansion for complex questions",
    ]
    for r in recs:
        story.append(Paragraph(f"• {r}", styles['BulletCustom']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>9.3 Repository</b>", styles['SubSection']))
    story.append(Paragraph(f"Main: {GITHUB_REPO}", styles['BodyTextCustom']))
    story.append(Paragraph(f"RAG System: {GITHUB_REPO}/blob/main/chromadb_rag_system.py", styles['BodyTextCustom']))
    story.append(Paragraph(f"Evaluation: {GITHUB_REPO}/blob/main/evaluate_chromadb_fast.py", styles['BodyTextCustom']))
    story.append(Paragraph(f"UI: {GITHUB_REPO}/blob/main/app_chromadb.py", styles['BodyTextCustom']))
    story.append(Paragraph(f"Docs: {GITHUB_REPO}/tree/main/docs", styles['BodyTextCustom']))
    
    # Build PDF
    print("📄 Generating PDF report...")
    doc.build(story)
    
    print(f"✅ PDF generated: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    return pdf_path

if __name__ == "__main__":
    generate_pdf()
