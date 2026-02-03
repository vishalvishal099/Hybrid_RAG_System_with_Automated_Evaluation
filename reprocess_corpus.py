"""
Reprocess Corpus with Improved Chunking Strategy
This will reload your existing corpus and re-chunk it with better logic
"""

import json
import sys
import re
from pathlib import Path
from tqdm import tqdm
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Import the ImprovedChunker directly (inline to avoid import issues)
exec(open('src/improved_chunking.py').read())
from src.improved_chunking import ImprovedChunker


def reprocess_corpus():
    """Reprocess the existing corpus with improved chunking"""
    
    # Load existing corpus
    corpus_path = Path("data/corpus.json")
    if not corpus_path.exists():
        print("❌ Corpus file not found. Please run data collection first.")
        return
    
    print("Loading existing corpus...")
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    print(f"✓ Loaded corpus with {len(corpus['documents'])} documents")
    print(f"  Old chunking: {corpus['metadata']['total_chunks']} chunks")
    
    # Initialize improved chunker
    chunker = ImprovedChunker(
        min_tokens=150,  # Increased from 200 for better granularity
        max_tokens=350,  # Decreased from 400 for more focused chunks
        overlap_tokens=50  # Kept the same
    )
    
    # Reprocess each document
    new_chunks = []
    chunk_id = 0
    documents_metadata = []
    
    print("\nReprocessing documents with improved chunking...")
    for doc in tqdm(corpus['documents'], desc="Processing"):
        # The original corpus has text scattered across chunks
        # We need to reconstruct the full text per document
        
        # Get all chunks for this document
        doc_chunks = [chunk for chunk in corpus['chunks'] 
                     if chunk['url'] == doc['url']]
        
        if not doc_chunks:
            continue
        
        # Reconstruct full text (this is approximate, but best we can do)
        # In reality, we'd want to re-fetch from Wikipedia, but let's work with what we have
        full_text = ' '.join([chunk['text'] for chunk in doc_chunks])
        
        # Apply improved chunking
        new_doc_chunks = chunker.chunk_with_metadata(
            text=full_text,
            title=doc['title'],
            url=doc['url']
        )
        
        # Add chunk IDs and finalize
        for chunk_data in new_doc_chunks:
            chunk_data['chunk_id'] = str(chunk_id)
            new_chunks.append(chunk_data)
            chunk_id += 1
        
        # Update document metadata
        documents_metadata.append({
            'url': doc['url'],
            'title': doc['title'],
            'word_count': doc.get('word_count', len(full_text.split())),
            'num_chunks': len(new_doc_chunks),
            'summary': doc.get('summary', '')
        })
    
    # Create new corpus structure
    new_corpus = {
        'metadata': {
            'total_urls': len(documents_metadata),
            'total_chunks': len(new_chunks),
            'chunking_strategy': 'improved_semantic',
            'min_tokens': 150,
            'max_tokens': 350,
            'overlap_tokens': 50,
            'improvements': [
                'Semantic sentence boundary detection',
                'Proper overlap handling',
                'Context preservation with section headers',
                'Paragraph-aware splitting',
                'Long sentence handling'
            ]
        },
        'documents': documents_metadata,
        'chunks': new_chunks
    }
    
    # Backup old corpus
    backup_path = Path("data/corpus_old_chunking.json")
    print(f"\n📦 Backing up old corpus to {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    # Save new corpus
    print(f"💾 Saving improved corpus to {corpus_path}")
    with open(corpus_path, 'w') as f:
        json.dump(new_corpus, f, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 CORPUS REPROCESSING COMPLETE")
    print("="*60)
    print(f"Documents: {new_corpus['metadata']['total_urls']}")
    print(f"Old chunks: {corpus['metadata']['total_chunks']}")
    print(f"New chunks: {new_corpus['metadata']['total_chunks']}")
    print(f"Change: {new_corpus['metadata']['total_chunks'] - corpus['metadata']['total_chunks']:+d} chunks")
    print(f"\nAverage tokens per chunk:")
    
    token_counts = [chunk['token_count'] for chunk in new_chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    print(f"  Average: {avg_tokens:.1f}")
    print(f"  Min: {min_tokens}")
    print(f"  Max: {max_tokens}")
    
    print("\n✅ Next steps:")
    print("1. Rebuild indexes: python src/indexing.py")
    print("2. Restart Streamlit: streamlit run app.py")
    

if __name__ == "__main__":
    reprocess_corpus()
