"""
Simplified ChromaDB indexing - using ChromaDB's built-in embeddings
This avoids sentence-transformers dependency conflicts
"""

import json
import time
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

print("=" * 80)
print("CHROMADB INDEXING - SIMPLIFIED APPROACH")
print("=" * 80)

# Initialize ChromaDB client
print("\n🔧 Initializing ChromaDB...")
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Use default sentence-transformers embedding (all-MiniLM-L6-v2)
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Delete existing collection if exists
collection_name = "wikipedia_chunks"
try:
    client.delete_collection(name=collection_name)
    print(f"✓ Deleted existing collection: {collection_name}")
except:
    pass

# Create collection
collection = client.create_collection(
    name=collection_name,
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"}
)
print(f"✓ Created collection: {collection_name}")

# Load corpus
print("\n📂 Loading corpus...")
with open("data/corpus.json", 'r') as f:
    corpus_data = json.load(f)

chunks = corpus_data['chunks']
print(f"✓ Loaded {len(chunks)} chunks")

# Prepare data
texts = [chunk['text'] for chunk in chunks]
ids = [f"chunk_{i}" for i in range(len(texts))]
metadatas = [
    {
        'chunk_id': str(chunk.get('chunk_id', i)),
        'title': chunk.get('title', 'Unknown'),
        'url': chunk.get('url', ''),
    }
    for i, chunk in enumerate(chunks)
]

# Add in batches
print(f"\n🔄 Adding {len(texts)} documents to ChromaDB...")
batch_size = 100
start_time = time.time()

for i in range(0, len(texts), batch_size):
    end_idx = min(i + batch_size, len(texts))
    batch_texts = texts[i:end_idx]
    batch_ids = ids[i:end_idx]
    batch_metadatas = metadatas[i:end_idx]
    
    collection.add(
        documents=batch_texts,
        ids=batch_ids,
        metadatas=batch_metadatas
    )
    
    if (end_idx) % 500 == 0 or end_idx == len(texts):
        elapsed = time.time() - start_time
        rate = end_idx / elapsed
        print(f"  Progress: {end_idx}/{len(texts)} ({rate:.1f} docs/sec)")

total_time = time.time() - start_time
print(f"\n✓ ChromaDB index built with {collection.count()} vectors in {total_time:.1f}s")

# Test retrieval
print("\n🧪 Testing retrieval...")
results = collection.query(
    query_texts=["What is artificial intelligence?"],
    n_results=5
)

print(f"✓ Retrieved {len(results['ids'][0])} results")
print(f"\nTop result:")
print(f"  Document: {results['documents'][0][0][:200]}...")
print(f"  Distance: {results['distances'][0][0]:.4f}")

# Save stats
stats = {
    'total_documents': collection.count(),
    'embedding_function': 'sentence-transformers/all-MiniLM-L6-v2 (default)',
    'indexing_time_seconds': total_time,
    'distance_metric': 'cosine'
}

Path("chroma_db").mkdir(exist_ok=True)
with open("chroma_db/stats.json", 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 80)
print("✅ CHROMADB INDEXING COMPLETE!")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  - Total documents: {collection.count()}")
print(f"  - Indexing time: {total_time:.1f}s")
print(f"  - Average rate: {len(texts)/total_time:.1f} docs/sec")
print(f"  - Storage: ./chroma_db/")
print(f"\n✓ ChromaDB is ready for use!")
