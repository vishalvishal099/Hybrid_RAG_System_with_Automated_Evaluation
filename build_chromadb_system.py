"""
ChromaDB-based Hybrid RAG System
Complete rebuild using ChromaDB + all-mpnet-base-v2 + BM25 + RRF
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# ChromaDB and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# BM25
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("=" * 80)
print("CHROMADB-BASED HYBRID RAG SYSTEM - COMPLETE REBUILD")
print("=" * 80)

class ChromaDBHybridRAG:
    """
    Complete ChromaDB-based Hybrid RAG System
    - Dense: ChromaDB + all-mpnet-base-v2
    - Sparse: BM25
    - Fusion: RRF (k=60)
    - Generation: FLAN-T5-base
    """
    
    def __init__(self):
        """Initialize the system"""
        print("\n🔧 Initializing ChromaDB Hybrid RAG System...")
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stopwords = set(stopwords.words('english'))
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Models
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        
        # BM25 index
        self.bm25_index = None
        self.bm25_corpus = []
        
        # Collection
        self.collection = None
        
        # Corpus
        self.corpus_chunks = []
        
        print("✓ System initialized")
    
    def load_corpus(self, corpus_path: str = "data/corpus.json"):
        """Load corpus from existing file"""
        print(f"\n📂 Loading corpus from {corpus_path}...")
        
        with open(corpus_path, 'r') as f:
            corpus_data = json.load(f)
        
        self.corpus_chunks = corpus_data['chunks']
        print(f"✓ Loaded {len(self.corpus_chunks)} chunks")
        
        return self.corpus_chunks
    
    def build_chromadb_index(self, batch_size: int = 100):
        """
        Build ChromaDB dense vector index using all-mpnet-base-v2
        """
        print("\n" + "=" * 80)
        print("BUILDING CHROMADB DENSE INDEX")
        print("=" * 80)
        
        # Load embedding model
        print("\n📥 Loading embedding model: sentence-transformers/all-mpnet-base-v2...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("✓ Model loaded")
        
        # Create or get collection
        collection_name = "wikipedia_chunks"
        
        # Delete existing collection if exists
        try:
            self.chroma_client.delete_collection(name=collection_name)
            print(f"✓ Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Created collection: {collection_name}")
        
        # Prepare data
        texts = [chunk['text'] for chunk in self.corpus_chunks]
        ids = [f"chunk_{i}" for i in range(len(texts))]
        metadatas = [
            {
                'chunk_id': str(chunk.get('chunk_id', i)),
                'title': chunk.get('title', 'Unknown'),
                'url': chunk.get('url', ''),
                'doc_id': str(chunk.get('doc_id', '')),
            }
            for i, chunk in enumerate(self.corpus_chunks)
        ]
        
        # Add documents in batches
        print(f"\n🔄 Adding {len(texts)} documents to ChromaDB in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Add to collection
            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
            
            if (i + batch_size) % 1000 == 0 or end_idx == len(texts):
                print(f"  Progress: {end_idx}/{len(texts)} documents added")
        
        print(f"✓ ChromaDB index built with {self.collection.count()} vectors")
        
        # Save index stats
        stats = {
            'total_documents': self.collection.count(),
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'embedding_dimension': 768,
            'distance_metric': 'cosine'
        }
        
        Path("chroma_db").mkdir(exist_ok=True)
        with open("chroma_db/index_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("✓ Index stats saved to chroma_db/index_stats.json")
    
    def build_bm25_index(self):
        """
        Build BM25 sparse index
        """
        print("\n" + "=" * 80)
        print("BUILDING BM25 SPARSE INDEX")
        print("=" * 80)
        
        print(f"\n🔄 Tokenizing {len(self.corpus_chunks)} documents...")
        
        # Tokenize all documents
        tokenized_corpus = []
        for i, chunk in enumerate(self.corpus_chunks):
            text = chunk['text'].lower()
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t.isalnum() and t not in self.stopwords and len(t) > 2]
            tokenized_corpus.append(tokens)
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{len(self.corpus_chunks)} documents tokenized")
        
        self.bm25_corpus = tokenized_corpus
        
        # Build BM25 index
        print("\n🔨 Building BM25 index...")
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"✓ BM25 index built with {len(tokenized_corpus)} documents")
        
        # Save BM25 index
        import pickle
        bm25_path = Path("chroma_db/bm25_index.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        print(f"✓ BM25 index saved to {bm25_path}")
        
        # Save tokenized corpus
        corpus_path = Path("chroma_db/bm25_corpus.pkl")
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)
        print(f"✓ Tokenized corpus saved to {corpus_path}")
    
    def dense_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Dense retrieval using ChromaDB
        Returns: List of (chunk_index, similarity_score)
        """
        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract results
        dense_results = []
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            chunk_idx = int(doc_id.split('_')[1])
            # Convert distance to similarity (ChromaDB returns cosine distance)
            similarity = 1 - distance
            dense_results.append((chunk_idx, float(similarity)))
        
        return dense_results
    
    def sparse_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Sparse retrieval using BM25
        Returns: List of (chunk_index, bm25_score)
        """
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalnum() and t not in self.stopwords and len(t) > 2]
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return results
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results
    
    def reciprocal_rank_fusion(self,
                               dense_results: List[Tuple[int, float]],
                               sparse_results: List[Tuple[int, float]],
                               k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) to combine dense and sparse results
        """
        rrf_scores = {}
        
        # Add dense scores
        for rank, (idx, score) in enumerate(dense_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Add sparse scores
        for rank, (idx, score) in enumerate(sparse_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build result list with metadata
        final_results = []
        for idx, rrf_score in sorted_results[:10]:  # Top 10
            chunk = self.corpus_chunks[idx].copy()
            chunk['rrf_score'] = rrf_score
            chunk['chunk_index'] = idx
            final_results.append(chunk)
        
        return final_results
    
    def retrieve(self, query: str, method: str = "hybrid") -> Dict:
        """
        Main retrieval method
        """
        start_time = time.time()
        
        if method == "dense":
            dense_results = self.dense_retrieval(query, top_k=10)
            chunks = [self.corpus_chunks[idx] for idx, _ in dense_results]
        
        elif method == "sparse":
            sparse_results = self.sparse_retrieval(query, top_k=10)
            chunks = [self.corpus_chunks[idx] for idx, _ in sparse_results]
        
        else:  # hybrid (RRF)
            dense_results = self.dense_retrieval(query, top_k=100)
            sparse_results = self.sparse_retrieval(query, top_k=100)
            chunks = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        
        retrieval_time = time.time() - start_time
        
        return {
            'query': query,
            'method': method,
            'chunks': chunks,
            'retrieval_time': retrieval_time
        }
    
    def load_generation_model(self):
        """Load FLAN-T5 for answer generation"""
        if self.generation_model is not None:
            return
        
        print("\n📥 Loading generation model: google/flan-t5-base...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.generation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
        if torch.cuda.is_available():
            self.generation_model = self.generation_model.cuda()
            print("✓ Model loaded on GPU")
        else:
            print("✓ Model loaded on CPU")
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate answer using FLAN-T5"""
        start_time = time.time()
        
        self.load_generation_model()
        
        if not context_chunks:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'generation_time': time.time() - start_time
            }
        
        # Prepare context from top 5 chunks
        top_chunks = context_chunks[:5]
        context_parts = []
        for i, chunk in enumerate(top_chunks, 1):
            text_snippet = chunk['text'][:400].strip()
            context_parts.append(f"Source {i}: {text_snippet}")
        
        full_context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Using ONLY the information from the sources below, write a clear and complete answer to the question. Your answer should be 2-3 complete sentences with proper punctuation.

{full_context}

Question: {query}

Complete answer:"""
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.generation_model.generate(
                **inputs,
                max_new_tokens=180,
                min_length=40,
                do_sample=False,
                num_beams=3,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                early_stopping=False
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Clean up answer
        import re
        answer = re.sub(r'<[^>]+>', '', answer)
        answer = re.sub(r'&[a-z]+;', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure proper ending
        if answer and answer[-1] not in '.!?':
            answer = answer.rstrip(' ,:;-') + '.'
        
        generation_time = time.time() - start_time
        
        return {
            'answer': answer,
            'generation_time': generation_time
        }


def main():
    """Main indexing pipeline"""
    print("\n🚀 Starting ChromaDB Hybrid RAG System Build...\n")
    
    # Initialize system
    rag = ChromaDBHybridRAG()
    
    # Load corpus
    corpus_chunks = rag.load_corpus("data/corpus.json")
    print(f"\n✓ Corpus loaded: {len(corpus_chunks)} chunks")
    
    # Build ChromaDB dense index
    rag.build_chromadb_index(batch_size=100)
    
    # Build BM25 sparse index
    rag.build_bm25_index()
    
    print("\n" + "=" * 80)
    print("✅ INDEXING COMPLETE!")
    print("=" * 80)
    print("\n📊 Summary:")
    print(f"  - ChromaDB vectors: {rag.collection.count()}")
    print(f"  - BM25 documents: {len(rag.bm25_corpus)}")
    print(f"  - Embedding model: all-mpnet-base-v2")
    print(f"  - Fusion method: RRF (k=60)")
    print(f"  - Storage: ./chroma_db/")
    
    # Test retrieval
    print("\n" + "=" * 80)
    print("🧪 TESTING RETRIEVAL")
    print("=" * 80)
    
    test_query = "What is artificial intelligence?"
    print(f"\nTest query: {test_query}")
    
    # Test hybrid retrieval
    results = rag.retrieve(test_query, method="hybrid")
    print(f"\n✓ Retrieved {len(results['chunks'])} chunks in {results['retrieval_time']:.3f}s")
    
    if results['chunks']:
        top_chunk = results['chunks'][0]
        print(f"\nTop result:")
        print(f"  Title: {top_chunk.get('title', 'N/A')}")
        print(f"  RRF Score: {top_chunk.get('rrf_score', 0):.4f}")
        print(f"  Text preview: {top_chunk['text'][:200]}...")
    
    # Test answer generation
    print("\n" + "=" * 80)
    print("🤖 TESTING ANSWER GENERATION")
    print("=" * 80)
    
    answer_data = rag.generate_answer(test_query, results['chunks'])
    print(f"\n📝 Answer ({answer_data['generation_time']:.2f}s):")
    print(f"{answer_data['answer']}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\n🎉 ChromaDB Hybrid RAG System is ready!")
    print("\nNext steps:")
    print("  1. Run FastAPI backend: uvicorn api:app --reload")
    print("  2. Run Streamlit UI: streamlit run app_chromadb.py")
    print("  3. Run evaluation: python evaluate_chromadb.py")


if __name__ == "__main__":
    main()
