"""
Complete ChromaDB Hybrid RAG System
Combines ChromaDB (dense) + BM25 (sparse) + RRF fusion + FLAN-T5 generation
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class ChromaDBHybridRAG:
    """
    Complete Hybrid RAG with ChromaDB + BM25 + RRF
    """
    
    def __init__(self, chroma_path="./chroma_db"):
        """Initialize the system"""
        print("🔧 Initializing ChromaDB Hybrid RAG...")
        
        self.chroma_path = Path(chroma_path)
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stopwords = set(stopwords.words('english'))
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(
            name="wikipedia_chunks",
            embedding_function=default_ef
        )
        
        # Load BM25
        with open(self.chroma_path / "bm25_index.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        with open(self.chroma_path / "bm25_corpus.pkl", 'rb') as f:
            self.bm25_corpus = pickle.load(f)
        
        # Load corpus
        with open("data/corpus.json", 'r') as f:
            corpus_data = json.load(f)
        self.corpus_chunks = corpus_data['chunks']
        
        # Generation model (lazy loaded)
        self.generation_model = None
        self.tokenizer = None
        
        print(f"✓ System initialized")
        print(f"  - ChromaDB vectors: {self.collection.count()}")
        print(f"  - BM25 documents: {len(self.bm25_corpus)}")
        print(f"  - Corpus chunks: {len(self.corpus_chunks)}")
    
    def dense_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Dense retrieval using ChromaDB"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        dense_results = []
        for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
            chunk_idx = int(doc_id.split('_')[1])
            similarity = 1 - distance  # Convert distance to similarity
            dense_results.append((chunk_idx, float(similarity)))
        
        return dense_results
    
    def sparse_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Sparse retrieval using BM25"""
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalnum() and t not in self.stopwords and len(t) > 2]
        
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results
    
    def reciprocal_rank_fusion(self,
                               dense_results: List[Tuple[int, float]],
                               sparse_results: List[Tuple[int, float]],
                               k: int = 60) -> List[Dict]:
        """RRF fusion"""
        rrf_scores = {}
        dense_scores_map = {idx: score for idx, score in dense_results}
        sparse_scores_map = {idx: score for idx, score in sparse_results}
        
        # Add dense scores
        for rank, (idx, score) in enumerate(dense_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Add sparse scores
        for rank, (idx, score) in enumerate(sparse_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Sort and build results
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for idx, rrf_score in sorted_results[:10]:
            chunk = self.corpus_chunks[idx].copy()
            chunk['rrf_score'] = rrf_score
            chunk['dense_score'] = dense_scores_map.get(idx, 0.0)
            chunk['sparse_score'] = sparse_scores_map.get(idx, 0.0)
            chunk['chunk_index'] = idx
            final_results.append(chunk)
        
        return final_results
    
    def retrieve(self, query: str, method: str = "hybrid") -> Dict:
        """Main retrieval method"""
        start_time = time.time()
        
        if method == "dense":
            dense_results = self.dense_retrieval(query, top_k=10)
            chunks = []
            for idx, score in dense_results:
                chunk = self.corpus_chunks[idx].copy()
                chunk['dense_score'] = score
                chunk['sparse_score'] = 0.0
                chunk['rrf_score'] = 0.0
                chunk['chunk_index'] = idx
                chunks.append(chunk)
        
        elif method == "sparse":
            sparse_results = self.sparse_retrieval(query, top_k=10)
            chunks = []
            for idx, score in sparse_results:
                chunk = self.corpus_chunks[idx].copy()
                chunk['dense_score'] = 0.0
                chunk['sparse_score'] = score
                chunk['rrf_score'] = 0.0
                chunk['chunk_index'] = idx
                chunks.append(chunk)
        
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
        """Load FLAN-T5 for generation"""
        if self.generation_model is not None:
            return
        
        print("\n📥 Loading generation model...")
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
        
        # Prepare context
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
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract confidence score from token probabilities
        if hasattr(outputs, 'scores') and outputs.scores:
            # Calculate average confidence across generated tokens
            token_probs = []
            for score in outputs.scores:
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(score[0], dim=-1)
                # Get max probability for each token
                max_prob = probs.max().item()
                token_probs.append(max_prob)
            
            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
        else:
            confidence = 0.0
        
        answer = self.tokenizer.decode(outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0], 
                                       skip_special_tokens=True).strip()
        
        # Clean up
        import re
        answer = re.sub(r'<[^>]+>', '', answer)
        answer = re.sub(r'&[a-z]+;', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        if answer and answer[-1] not in '.!?':
            answer = answer.rstrip(' ,:;-') + '.'
        
        generation_time = time.time() - start_time
        
        return {
            'answer': answer,
            'generation_time': generation_time,
            'confidence': round(confidence, 4)
        }
    
    def query(self, query: str, method: str = "hybrid") -> Dict:
        """End-to-end RAG pipeline"""
        retrieval_results = self.retrieve(query, method)
        answer_data = self.generate_answer(query, retrieval_results['chunks'])
        
        return {
            'query': query,
            'method': method,
            'answer': answer_data['answer'],
            'sources': retrieval_results['chunks'][:5],
            'retrieval_time': retrieval_results['retrieval_time'],
            'generation_time': answer_data['generation_time'],
            'total_time': retrieval_results['retrieval_time'] + answer_data['generation_time']
        }


# Test
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING CHROMADB HYBRID RAG SYSTEM")
    print("=" * 80)
    
    rag = ChromaDBHybridRAG()
    
    test_query = "What is artificial intelligence?"
    print(f"\n📝 Query: {test_query}")
    
    result = rag.query(test_query)
    
    print(f"\n✅ Answer ({result['total_time']:.2f}s):")
    print(f"{result['answer']}")
    
    print(f"\n📚 Top sources:")
    for i, source in enumerate(result['sources'][:3], 1):
        print(f"{i}. {source.get('title', 'N/A')} (RRF: {source.get('rrf_score', 0):.4f})")
