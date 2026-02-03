"""
Indexing Script - Build FAISS and BM25 Indexes
Creates dense and sparse indexes for hybrid retrieval
"""

import os
# Fix threading issues with transformers/tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import yaml

print("Starting indexing script...")
print("Loading dependencies (this may take 2-3 minutes on first run)...")

# Import heavy libraries with progress indication
print("  [1/3] Loading sentence-transformers...")
from sentence_transformers import SentenceTransformer

print("  [2/3] Loading rank-bm25...")
from rank_bm25 import BM25Okapi

print("  [3/3] Loading FAISS...")
import faiss

from tqdm import tqdm

print("✓ All dependencies loaded!")
print()


class IndexBuilder:
    """Build and save FAISS and BM25 indexes"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Loading embedding model...")
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        try:
            # Try loading from cache first
            self.embedding_model = SentenceTransformer(
                self.config['models']['embedding_model'],
                local_files_only=True,
                device='cpu'
            )
            print(f"✓ Loaded {self.config['models']['embedding_model']} from cache")
        except Exception as e:
            print(f"Cache load failed: {e}")
            print("Downloading model...")
            self.embedding_model = SentenceTransformer(
                self.config['models']['embedding_model'],
                device='cpu'
            )
            print(f"✓ Downloaded and loaded {self.config['models']['embedding_model']}")
        
    def load_corpus(self) -> Dict:
        """Load corpus from JSON"""
        corpus_path = Path(self.config['data']['corpus_file'])
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")
        
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        
        print(f"✓ Loaded {len(corpus['chunks'])} chunks")
        return corpus
    
    def build_faiss_index(self, chunks: List[Dict]) -> tuple:
        """
        Build FAISS dense index
        Returns: (index, embeddings)
        """
        print("\n" + "="*60)
        print("BUILDING FAISS INDEX (Dense Retrieval)")
        print("="*60)
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        print(f"Creating FAISS index (dimension={dimension})...")
        
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        
        print(f"✓ FAISS index built with {index.ntotal} vectors")
        
        return index, embeddings
    
    def build_bm25_index(self, chunks: List[Dict]) -> BM25Okapi:
        """
        Build BM25 sparse index
        """
        print("\n" + "="*60)
        print("BUILDING BM25 INDEX (Sparse Retrieval)")
        print("="*60)
        
        # Extract texts and tokenize
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Tokenizing {len(texts)} chunks...")
        tokenized_corpus = [text.lower().split() for text in tqdm(texts)]
        
        # Create BM25 index
        print("Creating BM25 index...")
        bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.config['retrieval']['sparse']['bm25_k1'],
            b=self.config['retrieval']['sparse']['bm25_b']
        )
        
        print(f"✓ BM25 index built")
        
        return bm25
    
    def save_indexes(self, faiss_index, embeddings: np.ndarray, 
                     bm25_index: BM25Okapi, chunks: List[Dict]):
        """Save indexes to disk"""
        print("\n" + "="*60)
        print("SAVING INDEXES")
        print("="*60)
        
        # Create index directory
        index_dir = Path("data/indexes")
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_dir = index_dir / "faiss_index"
        faiss_dir.mkdir(exist_ok=True)
        
        faiss_index_path = faiss_dir / "index.faiss"
        faiss.write_index(faiss_index, str(faiss_index_path))
        print(f"✓ Saved FAISS index to {faiss_index_path}")
        
        # Save embeddings
        embeddings_path = faiss_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"✓ Saved embeddings to {embeddings_path}")
        
        # Save chunk metadata
        chunks_path = faiss_dir / "chunks.json"
        with open(chunks_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        print(f"✓ Saved chunk metadata to {chunks_path}")
        
        # Save BM25 index
        bm25_path = index_dir / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        print(f"✓ Saved BM25 index to {bm25_path}")
        
        # Save BM25 chunks (same as FAISS chunks)
        bm25_chunks_path = index_dir / "bm25_chunks.json"
        with open(bm25_chunks_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        print(f"✓ Saved BM25 chunks to {bm25_chunks_path}")
        
        print("\n" + "="*60)
        print("INDEX BUILDING COMPLETE")
        print("="*60)
        print(f"Total chunks indexed: {len(chunks)}")
        print(f"FAISS index size: {faiss_index.ntotal} vectors")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print("="*60)


def main():
    """Main execution"""
    builder = IndexBuilder()
    
    # Load corpus
    corpus = builder.load_corpus()
    chunks = corpus['chunks']
    
    # Build FAISS index
    faiss_index, embeddings = builder.build_faiss_index(chunks)
    
    # Build BM25 index
    bm25_index = builder.build_bm25_index(chunks)
    
    # Save indexes
    builder.save_indexes(faiss_index, embeddings, bm25_index, chunks)
    
    print("\n✅ All indexes built and saved successfully!")
    print("\nNext steps:")
    print("1. Generate questions: python src/question_generation.py")
    print("2. Run evaluation: python src/evaluation/evaluation_pipeline.py")
    print("3. Launch UI: streamlit run app.py")


if __name__ == "__main__":
    main()
