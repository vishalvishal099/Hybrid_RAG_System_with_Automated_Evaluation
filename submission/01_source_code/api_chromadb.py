"""
FastAPI Backend for ChromaDB Hybrid RAG System
Provides REST API for RAG queries
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import time

from chromadb_rag_system import ChromaDBHybridRAG

# Initialize FastAPI app
app = FastAPI(
    title="ChromaDB Hybrid RAG API",
    description="REST API for Hybrid RAG system with ChromaDB + BM25 + RRF",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system (lazy loaded)
rag_system = None

def get_rag_system():
    """Lazy load RAG system"""
    global rag_system
    if rag_system is None:
        print("🔧 Loading RAG system...")
        rag_system = ChromaDBHybridRAG()
        print("✓ RAG system ready")
    return rag_system


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    method: Optional[str] = "hybrid"  # "dense", "sparse", or "hybrid"
    include_sources: Optional[bool] = True


class Source(BaseModel):
    title: str
    url: str
    text_preview: str
    rrf_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    method: str
    sources: Optional[List[Source]] = None
    retrieval_time: float
    generation_time: float
    total_time: float


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    get_rag_system()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ChromaDB Hybrid RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Submit a question to the RAG system",
            "/health": "GET - Check system health",
            "/stats": "GET - Get system statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_rag_system()
        return {
            "status": "healthy",
            "chromadb_vectors": rag.collection.count(),
            "bm25_documents": len(rag.bm25_corpus),
            "corpus_chunks": len(rag.corpus_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System unhealthy: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        rag = get_rag_system()
        return {
            "system": "ChromaDB Hybrid RAG",
            "dense_retrieval": {
                "backend": "ChromaDB",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "total_vectors": rag.collection.count()
            },
            "sparse_retrieval": {
                "backend": "BM25",
                "total_documents": len(rag.bm25_corpus)
            },
            "fusion": {
                "method": "Reciprocal Rank Fusion (RRF)",
                "k_parameter": 60
            },
            "generation": {
                "model": "google/flan-t5-base"
            },
            "corpus": {
                "total_chunks": len(rag.corpus_chunks)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a question to the RAG system
    
    Args:
        query: The question to answer
        method: Retrieval method - "dense", "sparse", or "hybrid" (default)
        include_sources: Whether to include source documents in response
    
    Returns:
        QueryResponse with answer and optional sources
    """
    try:
        rag = get_rag_system()
        
        # Validate method
        if request.method not in ["dense", "sparse", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{request.method}'. Must be 'dense', 'sparse', or 'hybrid'"
            )
        
        # Execute query
        result = rag.query(request.query, method=request.method)
        
        # Prepare response
        response = QueryResponse(
            query=result['query'],
            answer=result['answer'],
            method=result['method'],
            retrieval_time=result['retrieval_time'],
            generation_time=result['generation_time'],
            total_time=result['total_time']
        )
        
        # Add sources if requested
        if request.include_sources and result['sources']:
            response.sources = [
                Source(
                    title=source.get('title', 'Unknown'),
                    url=source.get('url', ''),
                    text_preview=source['text'][:200] + "...",
                    rrf_score=source.get('rrf_score')
                )
                for source in result['sources']
            ]
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    """
    Retrieve relevant documents without generation
    
    Args:
        query: The search query
        method: Retrieval method - "dense", "sparse", or "hybrid"
    
    Returns:
        Retrieved documents with scores
    """
    try:
        rag = get_rag_system()
        
        # Execute retrieval only
        result = rag.retrieve(request.query, method=request.method)
        
        # Prepare response
        sources = [
            {
                "title": chunk.get('title', 'Unknown'),
                "url": chunk.get('url', ''),
                "text_preview": chunk['text'][:300] + "...",
                "rrf_score": chunk.get('rrf_score'),
                "chunk_index": chunk.get('chunk_index')
            }
            for chunk in result['chunks']
        ]
        
        return {
            "query": result['query'],
            "method": result['method'],
            "sources": sources,
            "retrieval_time": result['retrieval_time'],
            "total_results": len(sources)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


# Run with: uvicorn api_chromadb:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("STARTING CHROMADB HYBRID RAG API")
    print("=" * 80)
    print("\n📡 Server will be available at:")
    print("  - Local: http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("  - Health: http://localhost:8000/health")
    print("\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
