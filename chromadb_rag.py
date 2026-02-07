"""
ChromaDB-based Hybrid RAG System
Combines ChromaDB (dense) + BM25 (sparse) with RRF fusion
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBHybridRAG:
    """
    Hybrid RAG system using ChromaDB for dense retrieval and BM25 for sparse retrieval
    """
    
    def __init__(self, config_path: str = "config_chromadb.yaml"):
        """Initialize the ChromaDB Hybrid RAG system"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Initializing ChromaDB Hybrid RAG System")
        
        # Initialize embedding model
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        
        # Initialize ChromaDB client
        self.chroma_client = None
        self.collection = None
        
        # Initialize BM25
        self.bm25_index = None
        self.bm25_corpus = []
        
        # Corpus data
        self.corpus_chunks = []
        
        logger.info("✓ ChromaDB Hybrid RAG System initialized")
    
    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        if self.embedding_model is None:
            model_name = self.config['embedding']['model_name']
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("✓ Embedding model loaded")
    
    def initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        persist_dir = Path(self.config['chromadb']['persist_directory'])
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {persist_dir}")
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        collection_name = self.config['chromadb']['collection_name']
        distance_metric = self.config['chromadb']['distance_metric']
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"✓ Loaded existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            logger.info(f"✓ Created new ChromaDB collection: {collection_name}")
    
    def load_corpus(self, corpus_path: str = None):
        """Load the processed corpus"""
        if corpus_path is None:
            corpus_path = self.config['data']['corpus_file']
        
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r') as f:
            corpus_data = json.load(f)
        
        self.corpus_chunks = corpus_data['chunks']
        logger.info(f"✓ Loaded {len(self.corpus_chunks)} chunks")
    
    def build_chromadb_index(self):
        """Build ChromaDB index with embeddings"""
        logger.info("\n" + "="*60)
        logger.info("BUILDING CHROMADB INDEX")
        logger.info("="*60)
        
        # Load embedding model
        self.load_embedding_model()
        
        # Check if collection already has data
        if self.collection.count() > 0:
            logger.info(f"Collection already contains {self.collection.count()} vectors")
            response = input("Do you want to rebuild the index? (yes/no): ").lower()
            if response == 'yes':
                logger.info("Clearing existing collection...")
                self.chroma_client.delete_collection(self.config['chromadb']['collection_name'])
                self.initialize_chromadb()
            else:
                logger.info("Using existing ChromaDB index")
                return
        
        # Prepare data for indexing
        texts = [chunk['text'] for chunk in self.corpus_chunks]
        ids = [str(i) for i in range(len(self.corpus_chunks))]
        
        # Prepare metadata
        metadatas = []
        for chunk in self.corpus_chunks:
            metadata = {
                'url': chunk.get('url', ''),
                'title': chunk.get('title', ''),
                'chunk_id': str(chunk.get('chunk_id', '')),
            }
            metadatas.append(metadata)
        
        # Generate embeddings in batches
        batch_size = self.config['embedding']['batch_size']
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings_list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=self.config['embedding']['normalize']
            )
            embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        logger.info(f"✓ Generated embeddings: shape {embeddings.shape}")
        
        # Add to ChromaDB in batches
        logger.info("Adding documents to ChromaDB...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing"):
            end_idx = min(i + batch_size, len(texts))
            
            self.collection.add(
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        logger.info(f"✓ ChromaDB index built with {self.collection.count()} vectors")
    
    def build_bm25_index(self):
        """Build BM25 index for sparse retrieval"""
        logger.info("\n" + "="*60)
        logger.info("BUILDING BM25 INDEX")
        logger.info("="*60)
        
        # Tokenize corpus
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        stopwords_set = set(stopwords.words('english'))
        
        logger.info("Tokenizing corpus for BM25...")
        self.bm25_corpus = []
        
        for chunk in tqdm(self.corpus_chunks, desc="Tokenizing"):
            tokens = word_tokenize(chunk['text'].lower())
            tokens = [t for t in tokens if t.isalnum() and t not in stopwords_set]
            self.bm25_corpus.append(tokens)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25_index = BM25Okapi(
            self.bm25_corpus,
            k1=self.config['bm25']['k1'],
            b=self.config['bm25']['b']
        )
        
        logger.info(f"✓ BM25 index built with {len(self.bm25_corpus)} documents")
    
    def dense_retrieval(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Dense retrieval using ChromaDB"""
        if top_k is None:
            top_k = self.config['retrieval']['dense']['top_k']
        
        # Encode query
        self.load_embedding_model()
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.config['embedding']['normalize']
        )
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Extract indices and distances
        indices = [int(id_) for id_ in results['ids'][0]]
        distances = results['distances'][0]
        
        # Convert distances to similarity scores (assuming cosine distance)
        similarities = [1 - d for d in distances]
        
        return list(zip(indices, similarities))
    
    def sparse_retrieval(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """Sparse retrieval using BM25"""
        if top_k is None:
            top_k = self.config['retrieval']['sparse']['top_k']
        
        # Tokenize query
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        stopwords_set = set(stopwords.words('english'))
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalnum() and t not in stopwords_set]
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF) to combine dense and sparse results
        
        RRF formula: RRF_score(d) = Σ 1 / (k + rank(d))
        """
        rrf_scores = {}
        
        # Add dense results
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Add sparse results
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def retrieve(self, query: str, method: str = "hybrid", top_k: int = None) -> List[Dict]:
        """
        Main retrieval method
        
        Args:
            query: User query
            method: "dense", "sparse", or "hybrid"
            top_k: Number of results to return (uses config if None)
        
        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = self.config['retrieval']['rrf']['final_top_n']
        
        start_time = time.time()
        
        if method == "dense":
            # Dense only
            results = self.dense_retrieval(query)
            final_results = results[:top_k]
            
        elif method == "sparse":
            # Sparse only
            results = self.sparse_retrieval(query)
            final_results = results[:top_k]
            
        else:  # hybrid
            # Get both
            dense_results = self.dense_retrieval(query)
            sparse_results = self.sparse_retrieval(query)
            
            # Apply RRF
            rrf_k = self.config['retrieval']['rrf']['k']
            final_results = self.reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=rrf_k
            )[:top_k]
        
        retrieval_time = time.time() - start_time
        
        # Format results
        formatted_results = []
        for idx, score in final_results:
            chunk = self.corpus_chunks[idx].copy()
            chunk['retrieval_score'] = float(score)
            chunk['retrieval_method'] = method
            formatted_results.append(chunk)
        
        logger.info(f"Retrieved {len(formatted_results)} chunks in {retrieval_time:.3f}s using {method}")
        
        return formatted_results


def main():
    """Example usage"""
    
    # Initialize system
    rag = ChromaDBHybridRAG()
    
    # Load corpus
    rag.load_corpus()
    
    # Initialize ChromaDB
    rag.initialize_chromadb()
    
    # Build indexes
    rag.build_chromadb_index()
    rag.build_bm25_index()
    
    # Test retrieval
    query = "What is artificial intelligence?"
    results = rag.retrieve(query, method="hybrid", top_k=5)
    
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['title']}")
        print(f"Score: {result['retrieval_score']:.4f}")
        print(f"Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()
