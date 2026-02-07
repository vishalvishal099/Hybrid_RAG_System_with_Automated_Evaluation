"""
Build BM25 index for ChromaDB system
"""

import json
import pickle
from pathlib import Path
import time
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print("=" * 80)
print("BUILDING BM25 INDEX")
print("=" * 80)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("\n📥 Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# Load corpus
print("\n📂 Loading corpus...")
with open("data/corpus.json", 'r') as f:
    corpus_data = json.load(f)

chunks = corpus_data['chunks']
print(f"✓ Loaded {len(chunks)} chunks")

# Tokenize
print(f"\n🔄 Tokenizing documents...")
start_time = time.time()

tokenized_corpus = []
for i, chunk in enumerate(chunks):
    text = chunk['text'].lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words and len(t) > 2]
    tokenized_corpus.append(tokens)
    
    if (i + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        print(f"  Progress: {i + 1}/{len(chunks)} ({rate:.1f} docs/sec)")

tokenize_time = time.time() - start_time
print(f"✓ Tokenization complete in {tokenize_time:.1f}s")

# Build BM25
print("\n🔨 Building BM25 index...")
start_time = time.time()
bm25_index = BM25Okapi(tokenized_corpus)
build_time = time.time() - start_time
print(f"✓ BM25 index built in {build_time:.1f}s")

# Save
output_dir = Path("chroma_db")
output_dir.mkdir(exist_ok=True)

bm25_path = output_dir / "bm25_index.pkl"
with open(bm25_path, 'wb') as f:
    pickle.dump(bm25_index, f)
print(f"✓ BM25 index saved to {bm25_path}")

corpus_path = output_dir / "bm25_corpus.pkl"
with open(corpus_path, 'wb') as f:
    pickle.dump(tokenized_corpus, f)
print(f"✓ Tokenized corpus saved to {corpus_path}")

# Save stats
stats = {
    'total_documents': len(tokenized_corpus),
    'tokenization_time_seconds': tokenize_time,
    'build_time_seconds': build_time,
    'average_tokens_per_doc': sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus)
}

with open(output_dir / "bm25_stats.json", 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 80)
print("✅ BM25 INDEX COMPLETE!")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  - Total documents: {len(tokenized_corpus)}")
print(f"  - Avg tokens/doc: {stats['average_tokens_per_doc']:.1f}")
print(f"  - Tokenization time: {tokenize_time:.1f}s")
print(f"  - Build time: {build_time:.1f}s")
print(f"  - Storage: {output_dir}/")
