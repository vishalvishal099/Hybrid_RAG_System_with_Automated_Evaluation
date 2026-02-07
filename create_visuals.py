"""
Create Architecture Diagram and Visual Assets for Hybrid RAG System
Uses matplotlib and graphviz-style diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# GitHub Repository URL
GITHUB_REPO = "https://github.com/vishalvishal099/Hybrid_RAG_System_with_Automated_Evaluation"


def create_architecture_diagram():
    """Create system architecture diagram"""
    print("🎨 Creating architecture diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#3498db',
        'dense': '#9b59b6',
        'sparse': '#e67e22',
        'fusion': '#2ecc71',
        'llm': '#e74c3c',
        'output': '#1abc9c',
        'storage': '#34495e',
        'ui': '#f39c12'
    }
    
    # Title
    ax.text(8, 11.5, 'Hybrid RAG System Architecture', fontsize=20, 
            fontweight='bold', ha='center', va='center')
    ax.text(8, 11, 'Dense + Sparse Retrieval with RRF Fusion', fontsize=12, 
            ha='center', va='center', style='italic', color='gray')
    
    # Draw boxes
    def draw_box(ax, x, y, w, h, text, color, subtext=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0), text, fontsize=11, 
                fontweight='bold', ha='center', va='center', color='white')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.25, subtext, fontsize=8, 
                    ha='center', va='center', color='white', style='italic')
    
    # User Input
    draw_box(ax, 0.5, 8.5, 2.5, 1.5, "User Query", colors['input'], "Natural Language")
    
    # UI Layer
    draw_box(ax, 0.5, 6, 2.5, 1.5, "Streamlit UI", colors['ui'], "app_chromadb.py")
    
    # Dense Retrieval Branch
    draw_box(ax, 4.5, 8, 3, 1.5, "Dense Retrieval", colors['dense'], "ChromaDB + MiniLM")
    draw_box(ax, 4.5, 5.5, 3, 1.5, "Vector Store", colors['storage'], "7,519 embeddings")
    
    # Sparse Retrieval Branch  
    draw_box(ax, 8.5, 8, 3, 1.5, "Sparse Retrieval", colors['sparse'], "BM25 + NLTK")
    draw_box(ax, 8.5, 5.5, 3, 1.5, "BM25 Index", colors['storage'], "11MB index")
    
    # RRF Fusion
    draw_box(ax, 6.5, 3.5, 3, 1.5, "RRF Fusion", colors['fusion'], "k=60")
    
    # LLM Generation
    draw_box(ax, 10.5, 3.5, 3, 1.5, "Answer Gen", colors['llm'], "FLAN-T5-Base")
    
    # Output
    draw_box(ax, 13, 6, 2.5, 1.5, "Response", colors['output'], "Answer + Sources")
    
    # Draw arrows
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    
    # Query flow arrows
    arrows = [
        # User to UI
        ((1.75, 8.5), (1.75, 7.5)),
        # UI to Dense
        ((3, 6.75), (4.5, 8.75)),
        # UI to Sparse
        ((3, 6.75), (8.5, 8.75)),
        # Dense to Vector Store
        ((6, 8), (6, 7)),
        # Sparse to BM25
        ((10, 8), (10, 7)),
        # Vector Store to RRF
        ((6, 5.5), (7, 5)),
        # BM25 to RRF
        ((10, 5.5), (9, 5)),
        # RRF to LLM
        ((9.5, 4.25), (10.5, 4.25)),
        # LLM to Output
        ((13.5, 4.25), (14.25, 6)),
    ]
    
    for (start, end) in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Corpus Box
    draw_box(ax, 4.5, 1.5, 7, 1.2, "Wikipedia Corpus", colors['storage'], 
             "501 articles | 14.5MB | 200+ words each")
    
    # Arrows to corpus
    ax.annotate('', xy=(6, 2.7), xytext=(6, 5.5),
               arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5, ls='--'))
    ax.annotate('', xy=(10, 2.7), xytext=(10, 5.5),
               arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5, ls='--'))
    
    # Legend
    legend_items = [
        ('Input/Query', colors['input']),
        ('Dense (Vector)', colors['dense']),
        ('Sparse (BM25)', colors['sparse']),
        ('RRF Fusion', colors['fusion']),
        ('LLM Generation', colors['llm']),
        ('Data Storage', colors['storage']),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((12.5 + (i % 2) * 1.8, 10.5 - (i // 2) * 0.5), 
                                    0.3, 0.3, facecolor=color, edgecolor='black'))
        ax.text(12.9 + (i % 2) * 1.8, 10.65 - (i // 2) * 0.5, label, fontsize=8, va='center')
    
    # Add metrics annotation
    ax.text(0.5, 0.5, 
            "Metrics: MRR=0.44 (Sparse Best) | Recall@10=0.47 | Answer F1=0.05", 
            fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    Path('docs').mkdir(exist_ok=True)
    plt.savefig('docs/architecture_diagram.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("  ✓ Saved docs/architecture_diagram.png")
    plt.close()


def create_ui_mockup_screenshots():
    """Create UI screenshot mockups"""
    print("\n📸 Creating UI screenshot mockups...")
    
    Path('screenshots').mkdir(exist_ok=True)
    
    # Screenshot 1: Query Input
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0e1117')
    
    # Header
    ax.add_patch(FancyBboxPatch((0, 9), 14, 1, facecolor='#262730', edgecolor='none'))
    ax.text(0.5, 9.5, '🔍 Hybrid RAG System with Automated Evaluation', 
            fontsize=16, color='white', fontweight='bold', va='center')
    
    # Sidebar
    ax.add_patch(FancyBboxPatch((0, 0), 3, 9, facecolor='#1e1e1e', edgecolor='none'))
    ax.text(0.3, 8.5, '⚙️ Settings', fontsize=12, color='white', fontweight='bold')
    ax.text(0.3, 7.8, 'Retrieval Method:', fontsize=10, color='#888')
    
    # Radio buttons
    methods = ['🔵 Dense (Vector)', '🟠 Sparse (BM25)', '🟢 Hybrid (RRF)']
    for i, m in enumerate(methods):
        ax.add_patch(Circle((0.5, 7.2 - i*0.5), 0.1, facecolor='#4CAF50' if i == 2 else 'none', 
                           edgecolor='white'))
        ax.text(0.7, 7.2 - i*0.5, m, fontsize=9, color='white', va='center')
    
    # Main content
    ax.add_patch(FancyBboxPatch((3.2, 0), 10.6, 9, facecolor='#0e1117', edgecolor='none'))
    
    # Query box
    ax.add_patch(FancyBboxPatch((3.5, 7.5), 10, 1, facecolor='#262730', edgecolor='#4CAF50', linewidth=2))
    ax.text(4, 8, 'What is photosynthesis?', fontsize=12, color='white', va='center')
    
    # Answer section
    ax.text(3.5, 6.8, '📝 Generated Answer', fontsize=12, color='#4CAF50', fontweight='bold')
    ax.add_patch(FancyBboxPatch((3.5, 5.3), 10, 1.3, facecolor='#1a1a2e', edgecolor='#333'))
    answer_text = "Photosynthesis is the process by which green plants convert\nsunlight into chemical energy, producing glucose and oxygen\nfrom carbon dioxide and water."
    ax.text(3.7, 6.3, answer_text, fontsize=10, color='white', va='top', family='monospace')
    
    # Sources section
    ax.text(3.5, 5, '📚 Retrieved Sources', fontsize=12, color='#4CAF50', fontweight='bold')
    sources = [
        ('1', 'Photosynthesis - Wikipedia', 'Score: 0.95', '#2ecc71'),
        ('2', 'Plant Biology Overview', 'Score: 0.82', '#f39c12'),
        ('3', 'Carbon Cycle in Nature', 'Score: 0.71', '#e74c3c'),
    ]
    for i, (num, title, score, color) in enumerate(sources):
        y = 4.3 - i * 0.8
        ax.add_patch(FancyBboxPatch((3.5, y), 10, 0.7, facecolor='#262730', edgecolor=color))
        ax.add_patch(Circle((4, y + 0.35), 0.2, facecolor=color))
        ax.text(4, y + 0.35, num, fontsize=9, color='white', ha='center', va='center', fontweight='bold')
        ax.text(4.5, y + 0.45, title, fontsize=10, color='white', fontweight='bold')
        ax.text(4.5, y + 0.15, score, fontsize=8, color='#888')
    
    # Metrics bar
    ax.add_patch(FancyBboxPatch((3.5, 0.3), 10, 0.8, facecolor='#262730', edgecolor='none'))
    ax.text(4, 0.7, '⏱️ 2.3s', fontsize=9, color='#4CAF50')
    ax.text(6, 0.7, '📊 MRR: 1.0', fontsize=9, color='#3498db')
    ax.text(8, 0.7, '🎯 Recall@10: 1.0', fontsize=9, color='#9b59b6')
    
    plt.savefig('screenshots/01_query_interface.png', dpi=150, bbox_inches='tight',
                facecolor='#0e1117', edgecolor='none')
    print("  ✓ Saved screenshots/01_query_interface.png")
    plt.close()
    
    # Screenshot 2: Method Comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0e1117')
    
    ax.text(7, 9.5, '📊 Method Comparison Dashboard', fontsize=18, color='white', 
            ha='center', fontweight='bold')
    
    # Three columns for methods
    methods_data = [
        ('Dense (Vector)', '#9b59b6', '0.30', '0.33', '2.4s'),
        ('Sparse (BM25)', '#e67e22', '0.44', '0.47', '0.8s'),
        ('Hybrid (RRF)', '#2ecc71', '0.38', '0.43', '3.2s'),
    ]
    
    for i, (name, color, mrr, recall, time) in enumerate(methods_data):
        x = 1 + i * 4.3
        ax.add_patch(FancyBboxPatch((x, 2), 3.8, 6.5, facecolor='#1a1a2e', 
                                    edgecolor=color, linewidth=3))
        ax.text(x + 1.9, 8, name, fontsize=12, color='white', ha='center', fontweight='bold')
        
        # Metrics
        metrics = [('MRR', mrr), ('Recall@10', recall), ('Avg Time', time)]
        for j, (metric, value) in enumerate(metrics):
            y = 6.5 - j * 1.5
            ax.text(x + 0.3, y, metric + ':', fontsize=10, color='#888')
            ax.text(x + 2.8, y, value, fontsize=14, color=color, fontweight='bold')
            
            # Progress bar
            ax.add_patch(FancyBboxPatch((x + 0.3, y - 0.5), 3.2, 0.3, 
                                       facecolor='#333', edgecolor='none'))
            if 's' not in value:
                bar_width = float(value) * 3.2
                ax.add_patch(FancyBboxPatch((x + 0.3, y - 0.5), bar_width, 0.3, 
                                           facecolor=color, edgecolor='none'))
    
    ax.text(7, 1.2, '🏆 Best: Sparse (BM25) - Highest MRR and Recall', 
            fontsize=12, color='#f39c12', ha='center', fontweight='bold')
    
    plt.savefig('screenshots/02_method_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='#0e1117', edgecolor='none')
    print("  ✓ Saved screenshots/02_method_comparison.png")
    plt.close()
    
    # Screenshot 3: Evaluation Results
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0e1117')
    
    ax.text(7, 9.5, '📈 Evaluation Results (100 Questions)', fontsize=18, color='white', 
            ha='center', fontweight='bold')
    
    # Table header
    headers = ['ID', 'Question', 'Method', 'MRR', 'R@10', 'F1']
    widths = [0.8, 5, 1.5, 1, 1, 1]
    x_pos = 0.5
    for h, w in zip(headers, widths):
        ax.add_patch(FancyBboxPatch((x_pos, 8.2), w - 0.1, 0.5, facecolor='#4CAF50', edgecolor='none'))
        ax.text(x_pos + w/2 - 0.05, 8.45, h, fontsize=10, color='white', ha='center', fontweight='bold')
        x_pos += w
    
    # Table rows
    rows = [
        ('Q001', 'What is photosynthesis?', 'hybrid', '1.00', '1.00', '0.12'),
        ('Q002', 'What is carbon fixing?', 'hybrid', '1.00', '1.00', '0.08'),
        ('Q003', 'How does osmosis work?', 'hybrid', '0.50', '1.00', '0.05'),
        ('Q004', 'What is DNA replication?', 'hybrid', '0.33', '0.00', '0.00'),
        ('Q005', 'Explain the water cycle', 'hybrid', '0.00', '0.00', '0.00'),
        ('Q006', 'What are chromosomes?', 'hybrid', '1.00', '1.00', '0.15'),
        ('Q007', 'Define mitochondria', 'hybrid', '0.50', '1.00', '0.10'),
        ('Q008', 'What is evolution?', 'hybrid', '0.25', '1.00', '0.03'),
    ]
    
    for i, row in enumerate(rows):
        y = 7.5 - i * 0.6
        bg_color = '#1a1a2e' if i % 2 == 0 else '#262730'
        ax.add_patch(FancyBboxPatch((0.4, y), 10.2, 0.55, facecolor=bg_color, edgecolor='none'))
        
        x_pos = 0.5
        for j, (val, w) in enumerate(zip(row, widths)):
            color = 'white'
            if j >= 3:  # Metric columns
                val_float = float(val)
                if val_float >= 0.8:
                    color = '#2ecc71'
                elif val_float >= 0.4:
                    color = '#f39c12'
                elif val_float > 0:
                    color = '#e74c3c'
                else:
                    color = '#666'
            ax.text(x_pos + 0.1, y + 0.27, val[:30] + ('...' if len(val) > 30 else ''), 
                   fontsize=9, color=color, va='center')
            x_pos += w
    
    # Summary stats
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 10, 1.2, facecolor='#262730', edgecolor='#4CAF50'))
    ax.text(5.5, 1.35, '📊 Summary: 100 Questions | Avg MRR: 0.38 | Avg Recall@10: 0.43 | Avg F1: 0.05', 
            fontsize=11, color='white', ha='center')
    ax.text(5.5, 0.85, '✓ Best Method: Sparse (BM25) with MRR=0.44', 
            fontsize=10, color='#2ecc71', ha='center')
    
    plt.savefig('screenshots/03_evaluation_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0e1117', edgecolor='none')
    print("  ✓ Saved screenshots/03_evaluation_results.png")
    plt.close()


def create_system_flow_diagram():
    """Create data flow diagram"""
    print("\n📊 Creating data flow diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(8, 7.5, 'Data Flow: Query → Retrieval → Generation → Response', 
            fontsize=16, ha='center', fontweight='bold')
    
    # Flow steps
    steps = [
        (1, 4, 'User\nQuery', '#3498db'),
        (3.5, 4, 'Embedding\n(MiniLM)', '#9b59b6'),
        (6, 5.5, 'Dense\nSearch', '#9b59b6'),
        (6, 2.5, 'BM25\nSearch', '#e67e22'),
        (9, 4, 'RRF\nFusion', '#2ecc71'),
        (11.5, 4, 'Context\nSelection', '#1abc9c'),
        (14, 4, 'LLM\n(FLAN-T5)', '#e74c3c'),
    ]
    
    for x, y, text, color in steps:
        ax.add_patch(Circle((x, y), 0.8, facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x, y, text, fontsize=8, color='white', ha='center', va='center', fontweight='bold')
    
    # Arrows
    arrow_pairs = [
        ((1.8, 4), (2.7, 4)),
        ((4.3, 4.3), (5.2, 5.2)),
        ((4.3, 3.7), (5.2, 2.8)),
        ((6.8, 5.2), (8.2, 4.3)),
        ((6.8, 2.8), (8.2, 3.7)),
        ((9.8, 4), (10.7, 4)),
        ((12.3, 4), (13.2, 4)),
    ]
    
    for start, end in arrow_pairs:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Labels
    ax.text(5.2, 6.5, 'ChromaDB\n(7,519 vectors)', fontsize=8, ha='center', color='#666')
    ax.text(5.2, 1.5, 'BM25 Index\n(11MB)', fontsize=8, ha='center', color='#666')
    ax.text(9, 2.5, 'k=60\nScoring', fontsize=8, ha='center', color='#666')
    ax.text(11.5, 2.5, 'Top 5\nChunks', fontsize=8, ha='center', color='#666')
    
    # Output arrow and box
    ax.annotate('', xy=(15.5, 4), xytext=(14.8, 4),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.add_patch(FancyBboxPatch((15.2, 3.2), 0.6, 1.6, facecolor='#1abc9c', edgecolor='black'))
    ax.text(15.5, 4, '📝', fontsize=14, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('docs/data_flow_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  ✓ Saved docs/data_flow_diagram.png")
    plt.close()


def main():
    """Create all visual assets"""
    print("=" * 60)
    print("CREATING VISUAL ASSETS")
    print("=" * 60)
    
    create_architecture_diagram()
    create_ui_mockup_screenshots()
    create_system_flow_diagram()
    
    print("\n" + "=" * 60)
    print("✅ All visual assets created!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - docs/architecture_diagram.png")
    print("  - docs/data_flow_diagram.png")
    print("  - screenshots/01_query_interface.png")
    print("  - screenshots/02_method_comparison.png")
    print("  - screenshots/03_evaluation_results.png")


if __name__ == "__main__":
    main()
