"""
Hybrid RAG System - Core Implementation
Combines Dense (FAISS) + Sparse (BM25) + RRF
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import yaml
from tqdm import tqdm

# Dense retrieval
from sentence_transformers import SentenceTransformer
import faiss

# Sparse retrieval
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# LLM Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class HybridRAGSystem:
    """
    Hybrid Retrieval-Augmented Generation System
    Combines dense vector retrieval (FAISS) + sparse keyword retrieval (BM25) + RRF
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG system"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stopwords = set(stopwords.words('english'))
        
        # Models
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        
        # Indexes
        self.faiss_index = None
        self.bm25_index = None
        
        # Data
        self.corpus_chunks = []
        self.chunk_metadata = []
        
        print("✓ Hybrid RAG System initialized")
    
    def load_corpus(self, corpus_path: str = None):
        """Load the processed corpus"""
        if corpus_path is None:
            corpus_path = self.config['data']['corpus_file']
        
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        
        self.corpus_chunks = corpus['chunks']
        self.chunk_metadata = corpus['documents']
        
        print(f"✓ Loaded {len(self.corpus_chunks)} chunks from {len(self.chunk_metadata)} documents")
    
    def build_dense_index(self):
        """
        Build dense vector index using sentence transformers + FAISS
        """
        print("\n" + "="*60)
        print("BUILDING DENSE VECTOR INDEX (FAISS)")
        print("="*60)
        
        # Load embedding model
        print(f"Loading embedding model: {self.config['models']['embedding_model']}")
        self.embedding_model = SentenceTransformer(
            self.config['models']['embedding_model']
        )
        
        # Extract texts
        texts = [chunk['text'] for chunk in self.corpus_chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        
        print(f"✓ FAISS index built with {self.faiss_index.ntotal} vectors")
        
        # Save index
        index_path = Path(self.config['paths']['vector_index'])
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_path))
        print(f"✓ Saved FAISS index to {index_path}")
    
    def build_sparse_index(self):
        """
        Build sparse BM25 index for keyword-based retrieval
        """
        print("\n" + "="*60)
        print("BUILDING SPARSE BM25 INDEX")
        print("="*60)
        
        # Tokenize corpus
        print("Tokenizing corpus for BM25...")
        tokenized_corpus = []
        
        for chunk in tqdm(self.corpus_chunks):
            tokens = word_tokenize(chunk['text'].lower())
            # Remove stopwords and short tokens
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
            tokenized_corpus.append(tokens)
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25_index = BM25Okapi(
            tokenized_corpus,
            k1=self.config['retrieval']['sparse']['bm25_k1'],
            b=self.config['retrieval']['sparse']['bm25_b']
        )
        
        print(f"✓ BM25 index built with {len(tokenized_corpus)} documents")
        
        # Save index
        index_path = Path(self.config['paths']['bm25_index'])
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25_index,
                'tokenized_corpus': tokenized_corpus
            }, f)
        
        print(f"✓ Saved BM25 index to {index_path}")
    
    def load_indexes(self):
        """Load pre-built indexes"""
        # Load FAISS index
        index_path = Path(self.config['paths']['vector_index'])
        if index_path.exists():
            print(f"Loading FAISS index from {index_path}...")
            self.faiss_index = faiss.read_index(str(index_path))
            print(f"✓ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        
        # Load BM25 index
        bm25_path = Path(self.config['paths']['bm25_index'])
        if bm25_path.exists():
            print(f"Loading BM25 index from {bm25_path}...")
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            print(f"✓ Loaded BM25 index")
        
        # Load embedding model if not loaded
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.config['models']['embedding_model']}")
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism to avoid threading issues
            
            try:
                # Try loading from cache first
                self.embedding_model = SentenceTransformer(
                    self.config['models']['embedding_model'],
                    local_files_only=True,
                    device='cpu'  # Force CPU to avoid CUDA issues
                )
                print("✓ Loaded embedding model from cache")
            except Exception as e:
                print(f"Loading from cache failed, downloading model: {e}")
                # Fall back to downloading if cache doesn't exist
                self.embedding_model = SentenceTransformer(
                    self.config['models']['embedding_model'],
                    device='cpu'
                )
                print("✓ Downloaded and loaded embedding model")
    
    def dense_retrieval(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Dense vector retrieval using FAISS
        Returns: List of (chunk_index, similarity_score)
        """
        if top_k is None:
            top_k = self.config['retrieval']['dense']['top_k']
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Return results
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], distances[0])]
        return results
    
    def sparse_retrieval(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Sparse BM25 retrieval
        Returns: List of (chunk_index, bm25_score)
        """
        if top_k is None:
            top_k = self.config['retrieval']['sparse']['top_k']
        
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t not in self.stopwords and len(t) > 2]
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return results
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results
    
    def reciprocal_rank_fusion(self, 
                               dense_results: List[Tuple[int, float]],
                               sparse_results: List[Tuple[int, float]],
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion
        RRF_score(d) = Σ 1/(k + rank_i(d))
        
        Args:
            dense_results: List of (chunk_index, score) from dense retrieval
            sparse_results: List of (chunk_index, score) from sparse retrieval
            k: RRF constant (default 60)
        
        Returns:
            List of (chunk_index, rrf_score) sorted by RRF score
        """
        rrf_scores = {}
        
        # Add dense retrieval scores
        for rank, (idx, _) in enumerate(dense_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
        
        # Add sparse retrieval scores
        for rank, (idx, _) in enumerate(sparse_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def retrieve(self, query: str, method: str = "hybrid") -> Dict:
        """
        Main retrieval method
        
        Args:
            query: User query
            method: "dense", "sparse", or "hybrid"
        
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        start_time = time.time()
        
        results = {
            'query': query,
            'method': method,
            'chunks': [],
            'retrieval_time': 0,
            'dense_results': [],
            'sparse_results': [],
            'rrf_results': []
        }
        
        if method == "dense":
            # Dense only
            dense_results = self.dense_retrieval(query)
            results['dense_results'] = dense_results
            
            top_n = self.config['retrieval']['rrf']['final_top_n']
            for idx, score in dense_results[:top_n]:
                chunk = self.corpus_chunks[idx].copy()
                chunk['retrieval_score'] = score
                chunk['retrieval_method'] = 'dense'
                results['chunks'].append(chunk)
        
        elif method == "sparse":
            # Sparse only
            sparse_results = self.sparse_retrieval(query)
            results['sparse_results'] = sparse_results
            
            top_n = self.config['retrieval']['rrf']['final_top_n']
            for idx, score in sparse_results[:top_n]:
                chunk = self.corpus_chunks[idx].copy()
                chunk['retrieval_score'] = score
                chunk['retrieval_method'] = 'sparse'
                results['chunks'].append(chunk)
        
        else:  # hybrid
            # Get both dense and sparse results
            dense_results = self.dense_retrieval(query)
            sparse_results = self.sparse_retrieval(query)
            
            results['dense_results'] = dense_results
            results['sparse_results'] = sparse_results
            
            # Apply RRF
            rrf_results = self.reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=self.config['retrieval']['rrf']['k']
            )
            results['rrf_results'] = rrf_results
            
            # Get top-N chunks
            top_n = self.config['retrieval']['rrf']['final_top_n']
            for idx, rrf_score in rrf_results[:top_n]:
                chunk = self.corpus_chunks[idx].copy()
                chunk['rrf_score'] = rrf_score
                chunk['retrieval_method'] = 'hybrid'
                
                # Add individual scores
                chunk['dense_score'] = next((s for i, s in dense_results if i == idx), 0)
                chunk['sparse_score'] = next((s for i, s in sparse_results if i == idx), 0)
                
                results['chunks'].append(chunk)
        
        results['retrieval_time'] = time.time() - start_time
        return results
    
    def load_generation_model(self):
        """Load the LLM for text generation"""
        if self.generation_model is None:
            print(f"Loading generation model: {self.config['models']['generation_model']}")
            try:
                # Try loading from cache first
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['models']['generation_model'],
                    local_files_only=True
                )
                self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config['models']['generation_model'],
                    local_files_only=True
                )
            except Exception as e:
                print(f"Loading from cache failed, downloading model: {e}")
                # Fall back to downloading if cache doesn't exist
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['models']['generation_model']
                )
                self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config['models']['generation_model']
                )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.generation_model = self.generation_model.cuda()
                print("✓ Model loaded on GPU")
            else:
                print("✓ Model loaded on CPU")
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """
        Generate answer using LLM with retrieved context
        """
        start_time = time.time()
        
        # Load model if needed
        self.load_generation_model()
        
        if not context_chunks:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'generation_time': time.time() - start_time
            }
        
        # Use top 3 chunks for focused, relevant context
        top_chunks = context_chunks[:3]
        context_parts = []
        for i, chunk in enumerate(top_chunks, 1):
            # Take first 500 chars from each chunk - enough for complete info
            text_snippet = chunk['text'][:500].strip()
            context_parts.append(f"{text_snippet}")
        
        full_context = "\n\n".join(context_parts)
        
        # Strict grounding prompt - forces model to use ONLY the provided context
        prompt = f"""Answer the question using ONLY the information provided in the context below. Do not use any external knowledge. If the context doesn't contain the answer, say "I cannot answer based on the provided information."

Context from Wikipedia:
{full_context}

Question: {query}

Answer based strictly on the context above:"""
        
        # Tokenize - keep moderate length for speed
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=768,  # Balanced for speed and context
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with balanced parameters - fast but complete
        with torch.no_grad():
            outputs = self.generation_model.generate(
                **inputs,
                max_new_tokens=150,  # Enough for 2-3 complete sentences
                min_length=20,
                do_sample=False,  # Greedy decoding for consistency
                num_beams=2,  # Minimal beams for speed
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Remove HTML entities like <br />, &nbsp;, etc.
        import re
        answer = re.sub(r'<[^>]+>', '', answer)  # Remove HTML tags
        answer = re.sub(r'&[a-z]+;', ' ', answer)  # Remove HTML entities
        answer = re.sub(r'\s+', ' ', answer).strip()  # Normalize whitespace
        
        # Check if answer says it cannot answer (respecting context limitation)
        cannot_answer_phrases = [
            "cannot answer",
            "don't have enough information",
            "not provided in the context",
            "not mentioned",
            "no information"
        ]
        
        # Validate answer quality and grounding
        answer_lower = answer.lower()
        is_valid_answer = (
            len(answer) >= 10 and 
            any(c.isalpha() for c in answer) and 
            answer.count('[') <= 3 and
            not any(phrase in answer_lower for phrase in cannot_answer_phrases)
        )
        
        if not is_valid_answer:
            # Answer is garbled or says "cannot answer", use extractive fallback from context
            # This ensures we ALWAYS return information from the retrieved Wikipedia pages
            best_text = " ".join([chunk['text'] for chunk in top_chunks[:2]])  # Use top 2 chunks
            sentences = best_text.split('.')
            
            # Look for sentences containing query keywords
            query_words = set(query.lower().split()) - {'what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
            relevant_sentences = []
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 40:  # Substantial sentence
                    sent_lower = sent.lower()
                    # Count keyword matches
                    matches = sum(1 for word in query_words if word in sent_lower)
                    if matches > 0:
                        relevant_sentences.append((matches, sent))
            
            if relevant_sentences:
                # Sort by number of matches and take best sentences
                relevant_sentences.sort(reverse=True, key=lambda x: x[0])
                answer = relevant_sentences[0][1] + "."
                # Add more sentences for complete answer (up to 3 sentences)
                if len(relevant_sentences) > 1:
                    answer += " " + relevant_sentences[1][1] + "."
                if len(relevant_sentences) > 2 and len(answer) < 200:
                    answer += " " + relevant_sentences[2][1] + "."
            else:
                # Fallback to first substantial sentences
                selected = []
                for sent in sentences:
                    if len(sent.strip()) > 40:
                        selected.append(sent.strip())
                        if len(selected) >= 2:
                            break
                if selected:
                    answer = ". ".join(selected) + "."
                else:
                    answer = best_text[:250] + "..."
        
        generation_time = time.time() - start_time
        
        return {
            'answer': answer,
            'generation_time': generation_time
        }
    
    def query(self, query: str, method: str = "hybrid") -> Dict:
        """
        End-to-end RAG pipeline
        
        Args:
            query: User question
            method: Retrieval method ("dense", "sparse", "hybrid")
        
        Returns:
            Complete response with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieval_results = self.retrieve(query, method)
        
        # Generate answer
        generation_results = self.generate_answer(query, retrieval_results['chunks'])
        
        # Combine results
        response = {
            'query': query,
            'answer': generation_results['answer'],
            'sources': [
                {
                    'title': chunk['title'],
                    'url': chunk['url'],
                    'text_preview': chunk['text'][:200] + "...",
                    'scores': {
                        'dense': chunk.get('dense_score', 0),
                        'sparse': chunk.get('sparse_score', 0),
                        'rrf': chunk.get('rrf_score', 0)
                    }
                }
                for chunk in retrieval_results['chunks']
            ],
            'metadata': {
                'method': method,
                'num_sources': len(retrieval_results['chunks']),
                'retrieval_time': retrieval_results['retrieval_time'],
                'generation_time': generation_results['generation_time'],
                'total_time': retrieval_results['retrieval_time'] + generation_results['generation_time']
            }
        }
        
        return response


def main():
    """Demo usage"""
    # Initialize system
    rag = HybridRAGSystem()
    
    # Load corpus
    rag.load_corpus()
    
    # Build indexes (comment out if already built)
    # rag.build_dense_index()
    # rag.build_sparse_index()
    
    # Or load existing indexes
    rag.load_indexes()
    
    # Example query
    query = "What is artificial intelligence?"
    print(f"\nQuery: {query}")
    
    response = rag.query(query, method="hybrid")
    
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(response['sources'], 1):
        print(f"{i}. {source['title']} - RRF: {source['scores']['rrf']:.4f}")
    
    print(f"\nRetrieval Time: {response['metadata']['retrieval_time']:.2f}s")
    print(f"Generation Time: {response['metadata']['generation_time']:.2f}s")
    print(f"Total Time: {response['metadata']['total_time']:.2f}s")


if __name__ == "__main__":
    main()
