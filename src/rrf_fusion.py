"""
Reciprocal Rank Fusion (RRF) Implementation
Combines Dense (FAISS) and Sparse (BM25) retrieval results
"""

from typing import List, Dict, Tuple
import numpy as np


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining multiple retrieval results
    Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
    where k=60 (standard parameter)
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion
        
        Args:
            k: RRF constant parameter (60 is standard, per spec)
        """
        self.k = k
    
    def fuse_rankings(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Fuse dense and sparse retrieval results using RRF
        
        Args:
            dense_results: List of (doc_id, score) from FAISS
            sparse_results: List of (doc_id, score) from BM25
        
        Returns:
            Fused results as List of (doc_id, rrf_score), sorted by score descending
        """
        rrf_scores = {}
        
        # Process dense results
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        
        # Process sparse results
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        
        # Sort by RRF score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused
    
    def fuse_multiple_rankings(
        self,
        rankings_list: List[List[Tuple[int, float]]]
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple retrieval result lists using RRF
        
        Args:
            rankings_list: List of result lists, each containing (doc_id, score) tuples
        
        Returns:
            Fused results as List of (doc_id, rrf_score), sorted by score descending
        """
        rrf_scores = {}
        
        # Process each ranking
        for results in rankings_list:
            for rank, (doc_id, _) in enumerate(results, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        
        # Sort by RRF score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused


def apply_rrf_to_results(
    faiss_indices: np.ndarray,
    faiss_scores: np.ndarray,
    bm25_indices: np.ndarray,
    bm25_scores: np.ndarray,
    k: int = 60,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Helper function to apply RRF to FAISS and BM25 results
    
    Args:
        faiss_indices: Array of document indices from FAISS
        faiss_scores: Array of similarity scores from FAISS
        bm25_indices: Array of document indices from BM25
        bm25_scores: Array of scores from BM25
        k: RRF constant (60)
        top_k: Number of top results to return
    
    Returns:
        Top-k fused results as List of (doc_id, rrf_score)
    """
    # Convert to list of tuples
    dense_results = list(zip(faiss_indices.tolist(), faiss_scores.tolist()))
    sparse_results = list(zip(bm25_indices.tolist(), bm25_scores.tolist()))
    
    # Apply RRF
    rrf = RRFFusion(k=k)
    fused = rrf.fuse_rankings(dense_results, sparse_results)
    
    # Return top-k
    return fused[:top_k]
