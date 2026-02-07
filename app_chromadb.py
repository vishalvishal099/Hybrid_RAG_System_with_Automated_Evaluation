"""
Streamlit UI for ChromaDB Hybrid RAG System
Provides interactive interface for querying the RAG system
"""

import streamlit as st
import time
from chromadb_rag_system import ChromaDBHybridRAG

# Page config
st.set_page_config(
    page_title="ChromaDB Hybrid RAG",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .source-card {
        background-color: #f0f2f6;
        border-left: 4px solid #2E86AB;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .source-title {
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 5px;
    }
    .source-text {
        color: #333;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


@st.cache_resource
def load_rag_system():
    """Load RAG system (cached)"""
    with st.spinner("🔧 Loading RAG system... This may take a minute..."):
        return ChromaDBHybridRAG()


# Header
st.markdown('<div class="main-header">🔍 ChromaDB Hybrid RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ChromaDB + BM25 + RRF + FLAN-T5</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Retrieval method
    method = st.selectbox(
        "Retrieval Method",
        ["hybrid", "dense", "sparse"],
        help="Dense: Vector similarity only | Sparse: BM25 only | Hybrid: RRF fusion"
    )
    
    # System info
    st.divider()
    st.subheader("📊 System Info")
    
    if st.button("🔄 Load System"):
        st.session_state.rag_system = load_rag_system()
        st.success("✓ System loaded!")
    
    if st.session_state.rag_system:
        st.metric("ChromaDB Vectors", f"{st.session_state.rag_system.collection.count():,}")
        st.metric("BM25 Documents", f"{len(st.session_state.rag_system.bm25_corpus):,}")
        st.metric("Total Chunks", f"{len(st.session_state.rag_system.corpus_chunks):,}")
    
    # Query history
    st.divider()
    st.subheader("📜 Recent Queries")
    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
            st.text(f"{i+1}. {q[:50]}...")
    else:
        st.info("No queries yet")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the capital of France?",
        key="query_input"
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.info("""
    **💡 Tips:**
    - Be specific in your questions
    - Try different retrieval methods
    - Check source documents below
    """)

# Clear functionality
if clear_button:
    st.session_state.query_history = []
    st.rerun()

# Search functionality
if search_button and query:
    # Load system if not loaded
    if st.session_state.rag_system is None:
        st.session_state.rag_system = load_rag_system()
    
    # Add to history
    st.session_state.query_history.append(query)
    
    # Execute query
    with st.spinner(f"🔍 Searching with {method} retrieval..."):
        start_time = time.time()
        result = st.session_state.rag_system.query(query, method=method)
        total_time = time.time() - start_time
    
    # Display results
    st.divider()
    
    # Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Method", result['method'].capitalize())
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Retrieval Time", f"{result['retrieval_time']:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Generation Time", f"{result['generation_time']:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Time", f"{result['total_time']:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Answer
    st.subheader("📝 Answer")
    st.markdown(f'<div class="answer-box"><strong>{result["answer"]}</strong></div>', unsafe_allow_html=True)
    
    # Sources
    st.subheader("📚 Source Documents")
    
    # Source display options
    show_all = st.checkbox("Show all sources", value=False)
    num_sources = len(result['sources']) if show_all else min(5, len(result['sources']))
    
    for i, source in enumerate(result['sources'][:num_sources]):
        with st.expander(f"📄 Source {i+1}: {source.get('title', 'Unknown')}", expanded=(i==0)):
            st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
            
            # Title and URL
            st.markdown(f'<div class="source-title">{source.get("title", "Unknown")}</div>', unsafe_allow_html=True)
            if source.get('url'):
                st.markdown(f"🔗 [{source['url']}]({source['url']})")
            
            # RRF Score
            if source.get('rrf_score'):
                st.caption(f"RRF Score: {source['rrf_score']:.4f}")
            
            # Text
            st.markdown(f'<div class="source-text">{source["text"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    if not show_all and len(result['sources']) > 5:
        st.info(f"Showing 5 of {len(result['sources'])} sources. Check 'Show all sources' to see more.")

elif search_button and not query:
    st.warning("⚠️ Please enter a question")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>
    🔍 <strong>ChromaDB Hybrid RAG System</strong> | 
    Dense (ChromaDB) + Sparse (BM25) + RRF + FLAN-T5
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions for first-time users
if not st.session_state.rag_system and not search_button:
    st.info("""
    👋 **Welcome to ChromaDB Hybrid RAG!**
    
    **How to use:**
    1. Click "🔄 Load System" in the sidebar (first time only)
    2. Enter your question in the search box
    3. Choose a retrieval method (hybrid recommended)
    4. Click "🔍 Search" to get your answer
    
    **Retrieval Methods:**
    - 🎯 **Hybrid**: Combines dense + sparse retrieval using RRF (best results)
    - 📊 **Dense**: Vector similarity search using ChromaDB
    - 📝 **Sparse**: Keyword matching using BM25
    """)
