"""
Semantic Chunking with Deduplication and Anchor Preservation
Implements 200-400 token chunks with 50-token overlap as per requirements
"""

import re
import hashlib
import tiktoken
from typing import List, Dict, Tuple, Set
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SemanticChunker:
    """
    Advanced chunking with:
    - 200-400 token chunks with 50-token overlap
    - Deduplication using content hashing
    - Anchor preservation (section headers, titles)
    - Paragraph-aware splitting
    - Enhanced metadata tracking
    """
    
    def __init__(
        self,
        min_tokens: int = 200,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.seen_hashes: Set[str] = set()  # For deduplication
        self.chunk_counter = 0
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
    
    def compute_hash(self, text: str) -> str:
        """Compute MD5 hash for deduplication"""
        normalized = ' '.join(text.lower().split())  # Normalize whitespace
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if chunk is duplicate"""
        chunk_hash = self.compute_hash(text)
        if chunk_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(chunk_hash)
        return False
    
    def extract_anchors(self, text: str) -> List[Dict]:
        """
        Extract section anchors (headers, titles) from text
        Returns list of {text, position, type}
        """
        anchors = []
        
        # Pattern 1: Markdown headers (== Header ==)
        for match in re.finditer(r'^={2,6}\s*([^=]+)\s*={2,6}$', text, re.MULTILINE):
            anchors.append({
                'text': match.group(1).strip(),
                'position': match.start(),
                'type': 'markdown_header'
            })
        
        # Pattern 2: Title case with colon (Introduction:)
        for match in re.finditer(r'^([A-Z][A-Za-z\s]{3,30}):$', text, re.MULTILINE):
            anchors.append({
                'text': match.group(1).strip(),
                'position': match.start(),
                'type': 'section_title'
            })
        
        # Pattern 3: Numbered sections (1. Introduction)
        for match in re.finditer(r'^(\d+\.\s+[A-Z][^\n]{3,50})$', text, re.MULTILINE):
            anchors.append({
                'text': match.group(1).strip(),
                'position': match.start(),
                'type': 'numbered_section'
            })
        
        return sorted(anchors, key=lambda x: x['position'])
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to regex if NLTK fails
            sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_overlapping_chunks(
        self,
        sentences: List[str],
        anchor_prefix: str = ""
    ) -> List[Dict]:
        """
        Create overlapping chunks from sentences
        Returns chunks with metadata
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Add anchor prefix tokens if present
        prefix_tokens = self.count_tokens(anchor_prefix) if anchor_prefix else 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max, split it by clauses
            if sentence_tokens > self.max_tokens - prefix_tokens:
                # Split long sentence by clauses
                clauses = self._split_by_clauses(sentence)
                for clause in clauses:
                    clause_tokens = self.count_tokens(clause)
                    if current_tokens + clause_tokens <= self.max_tokens - prefix_tokens:
                        current_chunk.append(clause)
                        current_tokens += clause_tokens
                    else:
                        # Finalize current chunk
                        if current_chunk and current_tokens >= self.min_tokens - prefix_tokens:
                            chunk_text = anchor_prefix + ' '.join(current_chunk)
                            if not self.is_duplicate(chunk_text):
                                chunks.append({
                                    'text': chunk_text.strip(),
                                    'tokens': current_tokens + prefix_tokens,
                                    'anchor': anchor_prefix.strip() if anchor_prefix else None,
                                    'chunk_id': self.chunk_counter
                                })
                                self.chunk_counter += 1
                        
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap(current_chunk)
                        current_chunk = [overlap_text, clause] if overlap_text else [clause]
                        current_tokens = self.count_tokens(' '.join(current_chunk))
                i += 1
                continue
            
            # Check if adding sentence exceeds max
            if current_tokens + sentence_tokens > self.max_tokens - prefix_tokens:
                # Finalize current chunk if it meets minimum
                if current_tokens >= self.min_tokens - prefix_tokens:
                    chunk_text = anchor_prefix + ' '.join(current_chunk)
                    if not self.is_duplicate(chunk_text):
                        chunks.append({
                            'text': chunk_text.strip(),
                            'tokens': current_tokens + prefix_tokens,
                            'anchor': anchor_prefix.strip() if anchor_prefix else None,
                            'chunk_id': self.chunk_counter
                        })
                        self.chunk_counter += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = self.count_tokens(' '.join(current_chunk)) if current_chunk else 0
                else:
                    # Chunk too small, keep adding
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                    i += 1
                    continue
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Finalize last chunk
        if current_chunk and current_tokens >= self.min_tokens - prefix_tokens:
            chunk_text = anchor_prefix + ' '.join(current_chunk)
            if not self.is_duplicate(chunk_text):
                chunks.append({
                    'text': chunk_text.strip(),
                    'tokens': current_tokens + prefix_tokens,
                    'anchor': anchor_prefix.strip() if anchor_prefix else None,
                    'chunk_id': self.chunk_counter
                })
                self.chunk_counter += 1
        
        return chunks
    
    def _get_overlap(self, current_chunk: List[str]) -> str:
        """Get overlap text from end of current chunk"""
        if not current_chunk:
            return ""
        
        overlap_text = ""
        overlap_tokens = 0
        
        # Take sentences from end until we have enough overlap
        for sentence in reversed(current_chunk):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap_tokens:
                overlap_text = sentence + " " + overlap_text
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_text.strip()
    
    def _split_by_clauses(self, sentence: str) -> List[str]:
        """Split long sentence by clauses (commas, semicolons, conjunctions)"""
        # Split on common clause boundaries
        clauses = re.split(r'[,;]\s+|\s+(?:and|or|but|yet|so)\s+', sentence)
        return [c.strip() for c in clauses if c.strip()]
    
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        title: str,
        url: str = None
    ) -> List[Dict]:
        """
        Chunk a document with full metadata
        
        Returns list of chunks with metadata:
        - text: chunk content
        - tokens: token count
        - doc_id: parent document ID
        - chunk_id: unique chunk identifier
        - position: position in document
        - title: document title
        - url: source URL
        - anchor: section header (if any)
        """
        # Extract anchors
        anchors = self.extract_anchors(text)
        
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        all_chunks = []
        current_anchor = None
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Check if this paragraph starts with an anchor
            para_start_pos = text.find(paragraph)
            for anchor in anchors:
                if abs(anchor['position'] - para_start_pos) < 10:
                    current_anchor = anchor['text']
                    break
            
            # Split paragraph into sentences
            sentences = self.split_into_sentences(paragraph)
            
            # Create chunks with anchor prefix
            anchor_prefix = f"[{current_anchor}] " if current_anchor else ""
            chunks = self.create_overlapping_chunks(sentences, anchor_prefix)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    'doc_id': doc_id,
                    'title': title,
                    'url': url,
                    'position': len(all_chunks),
                    'paragraph_index': para_idx
                })
                all_chunks.append(chunk)
        
        return all_chunks
    
    def reset_deduplication(self):
        """Reset deduplication cache (call between documents)"""
        self.seen_hashes.clear()


def process_corpus_with_semantic_chunking(
    corpus_data: List[Dict],
    min_tokens: int = 200,
    max_tokens: int = 400,
    overlap_tokens: int = 50
) -> Dict:
    """
    Process entire corpus with semantic chunking
    
    Args:
        corpus_data: List of documents with {title, text, url}
        min_tokens: Minimum tokens per chunk (200)
        max_tokens: Maximum tokens per chunk (400)
        overlap_tokens: Overlap between chunks (50)
    
    Returns:
        Dict with 'chunks' and 'documents' lists
    """
    chunker = SemanticChunker(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens
    )
    
    all_chunks = []
    doc_metadata = []
    
    for doc_idx, doc in enumerate(corpus_data):
        doc_id = f"doc_{doc_idx:04d}"
        
        # Chunk the document
        chunks = chunker.chunk_document(
            text=doc.get('text', ''),
            doc_id=doc_id,
            title=doc.get('title', 'Unknown'),
            url=doc.get('url')
        )
        
        all_chunks.extend(chunks)
        
        # Store document metadata
        doc_metadata.append({
            'doc_id': doc_id,
            'title': doc.get('title', 'Unknown'),
            'url': doc.get('url'),
            'num_chunks': len(chunks)
        })
        
        # Reset deduplication between documents
        chunker.reset_deduplication()
        
        if (doc_idx + 1) % 50 == 0:
            print(f"Processed {doc_idx + 1} documents, {len(all_chunks)} chunks total")
    
    print(f"\n✓ Chunking complete:")
    print(f"  - Total documents: {len(doc_metadata)}")
    print(f"  - Total chunks: {len(all_chunks)}")
    print(f"  - Average chunks per doc: {len(all_chunks) / len(doc_metadata):.1f}")
    
    return {
        'chunks': all_chunks,
        'documents': doc_metadata
    }
