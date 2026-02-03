"""
Streamlit User Interface for Hybrid RAG System
Display: query input, answer, sources, scores, response time
"""

# Disable tokenizers parallelism to avoid threading issues
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Fix for threading/atexit issues in Python 3.13
import sys
import threading
import atexit

# Prevent atexit registration errors during shutdown
_original_register = atexit.register
def _safe_register(func, *args, **kwargs):
    try:
        return _original_register(func, *args, **kwargs)
    except RuntimeError:
        pass
atexit.register = _safe_register

import streamlit as st
import time
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from src.rag_system import HybridRAGSystem


# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .source-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """Load and cache RAG system"""
    with st.spinner("Loading RAG system..."):
        rag = HybridRAGSystem()
        rag.load_corpus()
        rag.load_indexes()
    return rag


def display_sources(sources):
    """Display retrieved sources with scores - deduplicated by URL"""
    st.subheader("📚 Retrieved Sources")
    
    # Deduplicate by URL, keeping highest RRF score
    seen_urls = {}
    unique_sources = []
    
    for source in sources:
        url = source['url']
        if url not in seen_urls:
            seen_urls[url] = source
            unique_sources.append(source)
        else:
            # Keep the one with higher RRF score
            if source['scores']['rrf'] > seen_urls[url]['scores']['rrf']:
                # Replace in unique_sources list
                idx = unique_sources.index(seen_urls[url])
                unique_sources[idx] = source
                seen_urls[url] = source
    
    # Display unique sources
    for i, source in enumerate(unique_sources, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="source-card">
                    <h4>Source {i}: {source['title']}</h4>
                    <p><strong>URL:</strong> <a href="{source['url']}" target="_blank">{source['url']}</a></p>
                    <p><strong>Preview:</strong> {source['text_preview']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Dense Score", f"{source['scores']['dense']:.4f}")
                st.metric("Sparse Score", f"{source['scores']['sparse']:.4f}")
                st.metric("RRF Score", f"{source['scores']['rrf']:.4f}")


def display_scores_chart(sources):
    """Display scores as interactive chart"""
    st.subheader("📊 Retrieval Scores Visualization")
    
    # Prepare data
    titles = [s['title'][:30] + "..." if len(s['title']) > 30 else s['title'] for s in sources]
    dense_scores = [s['scores']['dense'] for s in sources]
    sparse_scores = [s['scores']['sparse'] for s in sources]
    rrf_scores = [s['scores']['rrf'] for s in sources]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Dense',
        x=titles,
        y=dense_scores,
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Bar(
        name='Sparse',
        x=titles,
        y=sparse_scores,
        marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        name='RRF',
        x=titles,
        y=rrf_scores,
        marker_color='#f39c12'
    ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title="Source",
        yaxis_title="Score",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_timing_breakdown(metadata):
    """Display timing breakdown pie chart"""
    st.subheader("⏱️ Time Breakdown")
    
    labels = ['Retrieval', 'Generation']
    values = [metadata['retrieval_time'], metadata['generation_time']]
    colors = ['#3498db', '#2ecc71']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3
    )])
    
    fig.update_layout(
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Hybrid RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Combining Dense Vector Retrieval + Sparse BM25 + Reciprocal Rank Fusion")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Hybrid+RAG", use_column_width=True)
        
        st.header("⚙️ Configuration")
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["hybrid", "dense", "sparse"],
            help="Choose retrieval method: hybrid (recommended), dense only, or sparse only"
        )
        
        st.markdown("---")
        
        st.header("📊 System Info")
        
        try:
            with open("data/corpus.json", 'r') as f:
                corpus = json.load(f)
                st.metric("Total Documents", corpus['metadata']['total_urls'])
                st.metric("Total Chunks", corpus['metadata']['total_chunks'])
        except:
            st.info("Corpus not loaded")
        
        st.markdown("---")
        
        st.header("📖 About")
        st.markdown("""
        This Hybrid RAG system combines:
        - **Dense Retrieval**: Sentence embeddings + FAISS
        - **Sparse Retrieval**: BM25 keyword matching
        - **RRF**: Reciprocal Rank Fusion
        - **Generation**: Flan-T5 LLM
        
        Built for educational purposes.
        """)
    
    # Main content
    # Initialize RAG system
    try:
        rag_system = load_rag_system()
        st.success("✓ RAG system loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}")
        st.info("Please run data collection and index building first:\n\n"
                "```\npython src/data_collection.py\npython -c 'from src.rag_system import HybridRAGSystem; "
                "rag = HybridRAGSystem(); rag.load_corpus(); rag.build_dense_index(); rag.build_sparse_index()'\n```")
        return
    
    # Query input
    st.header("💬 Ask a Question")
    
    # Initialize query in session state if not exists
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""
    if 'run_query' not in st.session_state:
        st.session_state.run_query = False
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    
    # Example questions
    with st.expander("💡 Example Questions - Click to use"):
        example_questions = [
            "What is artificial intelligence?",
            "Who invented the telephone?",
            "When was the Roman Empire founded?",
            "How does photosynthesis work?",
            "What are the main features of quantum computing?"
        ]
        
        cols = st.columns(len(example_questions))
        for i, (col, example) in enumerate(zip(cols, example_questions)):
            if col.button(f"📝 Example {i+1}", key=f"ex{i}", use_container_width=True):
                st.session_state.selected_example = example
                st.session_state.query_text = example
                st.session_state.run_query = True
                st.rerun()
    
    # Query input with two columns
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        # Use selected example if available, otherwise use session state
        default_value = st.session_state.get('selected_example', st.session_state.query_text)
        if st.session_state.selected_example:
            # Clear after using
            st.session_state.selected_example = None
        
        query = st.text_input(
            "Enter your question:",
            value=default_value,
            placeholder="e.g., What is artificial intelligence?",
            label_visibility="collapsed"
        )
    
    with col_button:
        st.write("")  # Add spacing
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Update session state
    if query != st.session_state.query_text:
        st.session_state.query_text = query
    
    # Process query when button clicked or Enter pressed
    should_process = search_button or st.session_state.run_query
    if st.session_state.run_query:
        st.session_state.run_query = False
    
    if query and should_process:
        with st.spinner("🔍 Searching and generating answer..."):
            start_time = time.time()
            
            try:
                # Get response
                response = rag_system.query(query, method=retrieval_method)
                
                # Display answer
                st.header("✨ Answer")
                
                # Clean and format the answer
                answer_text = response['answer'].strip()
                
                # Display in a nice box
                st.markdown(f"""
                <div style="background-color: #e8f4f8; padding: 1.5rem; border-radius: 10px; 
                            border-left: 5px solid #1f77b4; font-size: 1.1rem; line-height: 1.6;">
                    {answer_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Show raw answer in expander if needed for debugging
                if len(answer_text) < 20 or any(char in answer_text for char in ['[', ']', '|', '/']):
                    with st.expander("⚠️ Raw Model Output (for debugging)"):
                        st.code(answer_text)
                        st.warning("The model output seems unusual. This might indicate a model loading issue.")
                
                # Metrics row
                st.header("📈 Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Retrieval Method",
                        retrieval_method.upper(),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Sources Retrieved",
                        response['metadata']['num_sources'],
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Retrieval Time",
                        f"{response['metadata']['retrieval_time']:.3f}s",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Total Time",
                        f"{response['metadata']['total_time']:.3f}s",
                        delta=None
                    )
                
                # Two column layout for details
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    # Sources
                    display_sources(response['sources'])
                
                with col_right:
                    # Scores chart
                    display_scores_chart(response['sources'])
                    
                    # Timing breakdown
                    display_timing_breakdown(response['metadata'])
                
                # Full response JSON (expandable)
                with st.expander("🔍 View Full Response JSON"):
                    st.json(response)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Hybrid RAG System | Built with Streamlit, FAISS, BM25, and Flan-T5</p>
        <p>For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
