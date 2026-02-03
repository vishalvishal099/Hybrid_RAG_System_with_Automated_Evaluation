"""
Improved Chunking Strategy for Better RAG Performance
Implements semantic-aware chunking with proper context preservation
"""

import re
import tiktoken
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ImprovedChunker:
    """
    Advanced text chunking with:
    1. Semantic sentence boundary detection
    2. Proper overlap handling
    3. Context preservation (headers, titles)
    4. Paragraph-aware splitting
    """
    
    def __init__(self, min_tokens: int = 150, max_tokens: int = 350, 
                 overlap_tokens: int = 50, encoding_name: str = "cl100k_base"):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
    
    def extract_headers(self, text: str) -> List[tuple]:
        """Extract section headers and their positions"""
        # Common Wikipedia section patterns
        header_patterns = [
            r'^={2,6}\s*([^=]+)\s*={2,6}$',  # Markdown style
            r'^([A-Z][A-Za-z\s]+):$',         # Title case with colon
            r'^(\d+\.\s+[A-Z][^.]+)$'         # Numbered sections
        ]
        
        headers = []
        for i, line in enumerate(text.split('\n')):
            for pattern in header_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    headers.append((i, match.group(1).strip()))
                    break
        
        return headers
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or multiple spaces
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
    
    def create_overlapping_chunks(self, sentences: List[str], 
                                  context: str = "") -> List[str]:
        """
        Create chunks with proper overlapping windows
        
        Args:
            sentences: List of sentences to chunk
            context: Optional context (like section header) to prepend
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Add context if provided
        if context:
            context_tokens = self.count_tokens(context)
            if context_tokens < self.max_tokens // 2:  # Only if context is reasonable
                current_chunk.append(context)
                current_tokens = context_tokens
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence is too long, split it
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by clauses or length
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                i += 1
                continue
            
            # Check if adding sentence exceeds max_tokens
            if current_tokens + sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if self.count_tokens(chunk_text) >= self.min_tokens:
                        chunks.append(chunk_text)
                    
                    # Create overlap: keep last few sentences
                    overlap_chunk = []
                    overlap_tokens = 0
                    
                    for sent in reversed(current_chunk):
                        sent_tokens = self.count_tokens(sent)
                        if overlap_tokens + sent_tokens <= self.overlap_tokens:
                            overlap_chunk.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    
                    current_chunk = overlap_chunk
                    current_tokens = overlap_tokens
                else:
                    # Edge case: single sentence exceeds max but <= max_tokens
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_tokens:
                chunks.append(chunk_text)
            elif chunks:  # If too small, append to last chunk
                chunks[-1] = chunks[-1] + ' ' + chunk_text
        
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a very long sentence into smaller parts"""
        # Try splitting by clauses (commas, semicolons)
        parts = re.split(r'[,;]\s+', sentence)
        
        chunks = []
        current = []
        current_tokens = 0
        
        for part in parts:
            part_tokens = self.count_tokens(part)
            
            if current_tokens + part_tokens > self.max_tokens:
                if current:
                    chunks.append(' '.join(current))
                current = [part]
                current_tokens = part_tokens
            else:
                current.append(part)
                current_tokens += part_tokens
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def chunk_with_metadata(self, text: str, title: str = "", 
                           url: str = "") -> List[Dict[str, any]]:
        """
        Main chunking method with metadata preservation
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Extract headers/sections
        headers = self.extract_headers(text)
        
        # Split into paragraphs first
        paragraphs = self.split_into_paragraphs(text)
        
        all_chunks = []
        current_section = title  # Use title as initial context
        
        for paragraph in paragraphs:
            # Check if this paragraph is a header
            is_header = False
            for _, header_text in headers:
                if header_text in paragraph[:100]:  # Check first 100 chars
                    current_section = header_text
                    is_header = True
                    break
            
            if is_header:
                continue  # Skip headers themselves, they're used as context
            
            # Split paragraph into sentences
            sentences = self.split_into_sentences(paragraph)
            
            # Create chunks with current section as context
            context = f"Section: {current_section}. " if current_section else ""
            para_chunks = self.create_overlapping_chunks(sentences, context)
            
            all_chunks.extend(para_chunks)
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk_text in enumerate(all_chunks):
            chunk_docs.append({
                'text': chunk_text,
                'token_count': self.count_tokens(chunk_text),
                'chunk_index': i,
                'title': title,
                'url': url,
                'has_context': bool(current_section and current_section != title)
            })
        
        return chunk_docs
    
    def chunk_simple(self, text: str) -> List[str]:
        """Simple chunking without metadata (for compatibility)"""
        paragraphs = self.split_into_paragraphs(text)
        all_chunks = []
        
        for paragraph in paragraphs:
            sentences = self.split_into_sentences(paragraph)
            chunks = self.create_overlapping_chunks(sentences)
            all_chunks.extend(chunks)
        
        return all_chunks


# Example usage and testing
if __name__ == "__main__":
    # Test the chunker
    chunker = ImprovedChunker(min_tokens=150, max_tokens=350, overlap_tokens=50)
    
    sample_text = """
    Artificial Intelligence
    
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
    intelligence displayed by humans and animals. Leading AI textbooks define the field as the study 
    of "intelligent agents": any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    History
    
    The field of AI research was born at a workshop at Dartmouth College in 1956. The attendees became 
    the founders and leaders of AI research. They and their students produced programs that the press 
    described as "astonishing": computers were learning checkers strategies, solving word problems in 
    algebra, proving logical theorems and speaking English.
    
    Applications
    
    AI is used in various applications including healthcare, automotive, finance, and entertainment. 
    In healthcare, AI assists in diagnosis and treatment planning. Autonomous vehicles use AI for 
    navigation and decision-making. Financial institutions employ AI for fraud detection and trading.
    """
    
    chunks = chunker.chunk_with_metadata(sample_text, title="Artificial Intelligence", 
                                         url="https://en.wikipedia.org/wiki/Artificial_intelligence")
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({chunk['token_count']} tokens):")
        print(f"{chunk['text'][:200]}...")
