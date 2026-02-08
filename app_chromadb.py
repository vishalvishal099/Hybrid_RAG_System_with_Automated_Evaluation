"""
Streamlit UI for ChromaDB Hybrid RAG System
Provides interactive interface for querying the RAG system
"""

import streamlit as st
import time
import numpy as np
from chromadb_rag_system import ChromaDBHybridRAG

# Page config
st.set_page_config(
    page_title="Hybrid RAG System with Automated Evaluation",
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
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


@st.cache_resource
def load_rag_system():
    """Load RAG system (cached)"""
    with st.spinner("🔧 Loading RAG system... This may take a minute..."):
        return ChromaDBHybridRAG()


# Header
st.markdown('<div class="main-header">🔍 Hybrid RAG System with Automated Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ChromaDB + BM25 + RRF + FLAN-T5 | Dense + Sparse Retrieval</div>', unsafe_allow_html=True)

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

import plotly.express as px

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

    # Per-question breakdown
    st.subheader("📊 Question-by-Question Analysis")
    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            st.markdown(f"**Q{i}:** {q}")
            # Optionally show answer and metrics if available
            if st.session_state.rag_system:
                try:
                    res = st.session_state.rag_system.query(q, method=method)
                    st.write(f"Answer: {res['answer'][:100]}...")
                    st.write(f"Retrieval Time: {res.get('retrieval_time',0):.3f}s | Generation Time: {res.get('generation_time',0):.3f}s")
                    st.write(f"Top Score: {res['sources'][0].get('rrf_score',0):.4f}")
                except Exception as e:
                    st.write(f"Error: {e}")
    else:
        st.info("No queries yet for breakdown.")

    # Chunk visualization and comparison - will show after search
    if 'last_result' in st.session_state and st.session_state.last_result:
        # Create two columns for visualization and chunk comparison
        col_viz, col_chunks = st.columns([1, 1])
        
        with col_viz:
            st.subheader("📈 Chunk Score Visualization")
            try:
                res = st.session_state.last_result
                chunk_scores = []
                for idx, src in enumerate(res['sources'][:10]):  # Top 10 chunks
                    chunk_scores.append({
                        'Chunk': f"Chunk {idx+1}",
                        'Dense': src.get('dense_score', 0),
                        'Sparse': src.get('sparse_score', 0),
                        'RRF': src.get('rrf_score', 0)
                    })
                import pandas as pd
                df = pd.DataFrame(chunk_scores)
                # Melt for multi-series bar chart
                df_melted = df.melt(id_vars=['Chunk'], value_vars=['Dense', 'Sparse', 'RRF'], 
                                   var_name='Score Type', value_name='Score')
                fig = px.bar(df_melted, x='Chunk', y='Score', color='Score Type', 
                            title='Retrieval Scores by Chunk (Top 10)',
                            barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in chunk visualization: {e}")
        
        with col_chunks:
            st.subheader("🔍 Dense vs Sparse vs Hybrid Chunks")
            try:
                res = st.session_state.last_result
                # Sort by dense score (top 5)
                dense_sorted = sorted(res['sources'], key=lambda x: x.get('dense_score', 0), reverse=True)[:5]
                # Sort by sparse score (top 5)
                sparse_sorted = sorted(res['sources'], key=lambda x: x.get('sparse_score', 0), reverse=True)[:5]
                # Sort by RRF/hybrid score (top 5)
                hybrid_sorted = sorted(res['sources'], key=lambda x: x.get('rrf_score', 0), reverse=True)[:5]
                
                # Show comparison tabs
                tab1, tab2, tab3 = st.tabs(["🎯 Dense Top 5", "📝 Sparse Top 5", "🔀 Hybrid Top 5"])
                
                with tab1:
                    st.caption("Top 5 chunks by Dense (Vector Similarity)")
                    for idx, src in enumerate(dense_sorted, 1):
                        with st.expander(f"#{idx} - Score: {src.get('dense_score', 0):.4f}", expanded=(idx==1)):
                            st.markdown(f"**{src.get('title', 'Unknown')}**")
                            st.caption(f"🔗 [Source]({src.get('url', '#')})")
                            st.text(src['text'][:200] + "..." if len(src['text']) > 200 else src['text'])
                            st.caption(f"Dense: {src.get('dense_score', 0):.4f} | Sparse: {src.get('sparse_score', 0):.4f} | RRF: {src.get('rrf_score', 0):.4f}")
                
                with tab2:
                    st.caption("Top 5 chunks by Sparse (BM25 Keyword)")
                    for idx, src in enumerate(sparse_sorted, 1):
                        with st.expander(f"#{idx} - Score: {src.get('sparse_score', 0):.4f}", expanded=(idx==1)):
                            st.markdown(f"**{src.get('title', 'Unknown')}**")
                            st.caption(f"🔗 [Source]({src.get('url', '#')})")
                            st.text(src['text'][:200] + "..." if len(src['text']) > 200 else src['text'])
                            st.caption(f"Dense: {src.get('dense_score', 0):.4f} | Sparse: {src.get('sparse_score', 0):.4f} | RRF: {src.get('rrf_score', 0):.4f}")
                
                with tab3:
                    st.caption("Top 5 chunks by Hybrid (RRF Fusion)")
                    for idx, src in enumerate(hybrid_sorted, 1):
                        with st.expander(f"#{idx} - Score: {src.get('rrf_score', 0):.4f}", expanded=(idx==1)):
                            st.markdown(f"**{src.get('title', 'Unknown')}**")
                            st.caption(f"🔗 [Source]({src.get('url', '#')})")
                            st.text(src['text'][:200] + "..." if len(src['text']) > 200 else src['text'])
                            st.caption(f"Dense: {src.get('dense_score', 0):.4f} | Sparse: {src.get('sparse_score', 0):.4f} | RRF: {src.get('rrf_score', 0):.4f}")
                            
            except Exception as e:
                st.error(f"Error showing chunks: {e}")
    else:
        st.subheader("📈 Chunk Score Visualization")
        st.info("🔍 Search for a query to see chunk score visualization and retrieved chunks")

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
    
    # Store result for visualization
    st.session_state.last_result = result
    
    # Display results
    st.divider()
    
    # Primary Metrics Row
    st.subheader("⚡ Performance Metrics")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Method", result['method'].capitalize())
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Retrieval Time", f"{result['retrieval_time']:.3f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Generation Time", f"{result['generation_time']:.3f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Time", f"{result['total_time']:.3f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Retrieval Scores Row
    if result['sources']:
        st.subheader("📊 Retrieval Scores")
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        # Calculate average scores
        sources = result['sources']
        avg_dense_score = sum(s.get('dense_score', 0) for s in sources) / len(sources) if sources else 0
        avg_sparse_score = sum(s.get('sparse_score', 0) for s in sources) / len(sources) if sources else 0
        avg_rrf_score = sum(s.get('rrf_score', 0) for s in sources) / len(sources) if sources else 0
        
        # Score-based quality metrics (since we don't have ground truth in UI)
        # Top score indicates best match quality
        top_score = sources[0].get('rrf_score', 0) if result['method'] == 'hybrid' else \
                   sources[0].get('dense_score', 0) if result['method'] == 'dense' else \
                   sources[0].get('sparse_score', 0)
        
        # Score distribution shows retrieval confidence
        scores = [s.get('rrf_score', 0) for s in sources] if result['method'] == 'hybrid' else \
                [s.get('dense_score', 0) for s in sources] if result['method'] == 'dense' else \
                [s.get('sparse_score', 0) for s in sources]
        score_std = np.std(scores) if len(scores) > 1 else 0
        
        with col_s1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Dense Score", f"{avg_dense_score:.4f}")
            st.caption("Vector similarity")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_s2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Sparse Score", f"{avg_sparse_score:.4f}")
            st.caption("BM25 score")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_s3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Hybrid Score", f"{avg_rrf_score:.4f}")
            st.caption("RRF fusion")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_s4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Top Score", f"{top_score:.4f}")
            st.caption("Best match")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_s5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Score Std Dev", f"{score_std:.4f}")
            st.caption(f"{len(sources)} docs")
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
            
            # Scores
            score_cols = st.columns(3)
            with score_cols[0]:
                if source.get('dense_score') is not None:
                    st.caption(f"🎯 Dense Score: {source['dense_score']:.4f}")
            with score_cols[1]:
                if source.get('sparse_score') is not None:
                    st.caption(f"📝 Sparse Score: {source['sparse_score']:.4f}")
            with score_cols[2]:
                if source.get('rrf_score') is not None:
                    st.caption(f"🔀 RRF Score: {source['rrf_score']:.4f}")
            
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
    🔍 <strong>Hybrid RAG System with Automated Evaluation</strong> | 
    Dense (ChromaDB) + Sparse (BM25) + RRF Fusion + FLAN-T5 Generation
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions for first-time users
if not st.session_state.rag_system and not search_button:
    st.info("""
    👋 **Welcome to Hybrid RAG System with Automated Evaluation!**
    
    **How to use:**
    1. Click "🔄 Load System" in the sidebar (first time only)
    2. Enter your question in the search box
    3. Choose a retrieval method (hybrid recommended)
    4. Click "🔍 Search" to get your answer
    
    **Retrieval Methods:**
    - 🎯 **Hybrid**: Combines dense + sparse retrieval using RRF (best results)
    - 📊 **Dense**: Vector similarity search using ChromaDB
    - 📝 **Sparse**: Keyword matching using BM25
    
    **Metrics Explained:**
    - **Dense Score**: Cosine similarity from vector embeddings (0-1)
    - **Sparse Score**: BM25 keyword matching score
    - **Hybrid Score**: RRF (Reciprocal Rank Fusion) combined score
    - **Top Score**: Highest relevance score of retrieved documents
    - **Score Std Dev**: Score distribution (low = consistent quality, high = varied quality)
    """)
