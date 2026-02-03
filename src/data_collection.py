"""
Wikipedia Data Collection Script
Collects 200 fixed URLs + 300 random URLs, cleans and chunks text
"""

import json
import random
import re
import time
from typing import List, Dict, Tuple
from pathlib import Path
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import tiktoken
from tqdm import tqdm
import yaml


class WikipediaDataCollector:
    """Collect and process Wikipedia articles"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='HybridRAG/1.0 (Educational Project)'
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def generate_diverse_fixed_urls(self, count: int = 200) -> List[str]:
        """
        Generate 200 diverse Wikipedia URLs across multiple categories
        These will be stored in fixed_urls.json and remain constant
        """
        categories = {
            'Science': ['Physics', 'Chemistry', 'Biology', 'Astronomy', 'Geology'],
            'Technology': ['Artificial_intelligence', 'Computer_science', 'Robotics', 
                          'Internet', 'Quantum_computing'],
            'History': ['Ancient_Egypt', 'Roman_Empire', 'World_War_II', 
                       'Renaissance', 'Industrial_Revolution'],
            'Geography': ['Mount_Everest', 'Amazon_rainforest', 'Sahara', 
                         'Great_Barrier_Reef', 'Antarctica'],
            'Arts': ['Leonardo_da_Vinci', 'Vincent_van_Gogh', 'Shakespeare', 
                    'Classical_music', 'Renaissance_art'],
            'Sports': ['Olympic_Games', 'FIFA_World_Cup', 'Cricket', 
                      'Basketball', 'Tennis'],
            'Philosophy': ['Socrates', 'Plato', 'Aristotle', 'Ethics', 'Metaphysics'],
            'Literature': ['Homer', 'Jane_Austen', 'Charles_Dickens', 
                          'Leo_Tolstoy', 'Gabriel_García_Márquez'],
            'Mathematics': ['Calculus', 'Linear_algebra', 'Number_theory', 
                           'Topology', 'Statistics'],
            'Medicine': ['Anatomy', 'Genetics', 'Immunology', 'Neuroscience', 
                        'Pharmacology'],
        }
        
        # Seed articles for each category
        seed_titles = []
        for category, topics in categories.items():
            seed_titles.extend(topics)
        
        collected_urls = []
        visited = set()
        
        print(f"Generating {count} diverse Wikipedia URLs...")
        
        # Collect from seed articles
        for title in tqdm(seed_titles, desc="Processing seed articles"):
            if len(collected_urls) >= count:
                break
            
            page = self.wiki.page(title)
            if page.exists():
                word_count = len(page.text.split())
                if word_count >= self.config['data']['min_words_per_page']:
                    url = page.fullurl
                    if url not in visited:
                        collected_urls.append(url)
                        visited.add(url)
                        
                        # Get linked articles
                        links = list(page.links.keys())
                        random.shuffle(links)
                        
                        for link_title in links[:5]:  # Check up to 5 links per article
                            if len(collected_urls) >= count:
                                break
                            
                            link_page = self.wiki.page(link_title)
                            if link_page.exists():
                                link_word_count = len(link_page.text.split())
                                if link_word_count >= self.config['data']['min_words_per_page']:
                                    link_url = link_page.fullurl
                                    if link_url not in visited:
                                        collected_urls.append(link_url)
                                        visited.add(link_url)
            
            time.sleep(0.1)  # Rate limiting
        
        # If we need more, use random articles
        while len(collected_urls) < count:
            try:
                # Get random article
                response = requests.get(
                    'https://en.wikipedia.org/wiki/Special:Random',
                    allow_redirects=True
                )
                url = response.url
                
                if url not in visited:
                    # Extract title from URL
                    title = url.split('/wiki/')[-1]
                    page = self.wiki.page(title)
                    
                    if page.exists():
                        word_count = len(page.text.split())
                        if word_count >= self.config['data']['min_words_per_page']:
                            collected_urls.append(url)
                            visited.add(url)
                            print(f"Collected {len(collected_urls)}/{count} URLs")
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Error getting random article: {e}")
                continue
        
        return collected_urls[:count]
    
    def collect_random_urls(self, count: int = 300, existing_urls: set = None) -> List[str]:
        """
        Collect random Wikipedia URLs (changes each run)
        NO TIMEOUT - Will keep trying until all URLs are collected
        """
        if existing_urls is None:
            existing_urls = set()
        
        collected_urls = []
        attempts = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        print(f"Collecting {count} random Wikipedia URLs...")
        print("NO TIMEOUT - Will continue until all URLs are collected!")
        print("This may take 15-30 minutes. Please be patient...")
        
        with tqdm(total=count) as pbar:
            while len(collected_urls) < count:  # Removed max_attempts constraint
                attempts += 1
                
                # Check for too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n⚠️  Too many consecutive failures. Pausing for 10 seconds...")
                    time.sleep(10)
                    consecutive_failures = 0
                
                try:
                    response = requests.get(
                        'https://en.wikipedia.org/wiki/Special:Random',
                        allow_redirects=True,
                        timeout=15,
                        headers={'User-Agent': 'HybridRAG/1.0 (Educational Project)'}
                    )
                    url = response.url
                    
                    if url not in existing_urls and url not in collected_urls:
                        title = url.split('/wiki/')[-1]
                        
                        # Quick check before full page load
                        try:
                            page = self.wiki.page(title)
                            
                            if page.exists():
                                # Use summary first (faster than full text)
                                text = page.summary if hasattr(page, 'summary') else page.text
                                word_count = len(text.split())
                                
                                if word_count >= self.config['data']['min_words_per_page']:
                                    collected_urls.append(url)
                                    existing_urls.add(url)
                                    pbar.update(1)
                                    consecutive_failures = 0  # Reset on success
                        except Exception:
                            consecutive_failures += 1
                            continue
                    
                    # Rate limiting - be nice to Wikipedia
                    time.sleep(0.2)
                    
                except requests.Timeout:
                    consecutive_failures += 1
                    time.sleep(1)
                    continue
                except requests.RequestException:
                    consecutive_failures += 1
                    time.sleep(1)
                    continue
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Interrupted! Collected {len(collected_urls)}/{count} URLs so far.")
                    return collected_urls
                except Exception as e:
                    consecutive_failures += 1
                    continue
        
        # Success - all URLs collected!
        print(f"\n✓ Successfully collected all {len(collected_urls)} random URLs after {attempts} attempts")
        
        return collected_urls
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia text"""
        # Remove citations [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, min_tokens: int, max_tokens: int, 
                   overlap_tokens: int) -> List[str]:
        """
        Chunk text into overlapping segments
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if self.count_tokens(chunk_text) >= min_tokens:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.count_tokens(chunk_text) >= min_tokens:
                chunks.append(chunk_text)
        
        return chunks
    
    def process_url(self, url: str, chunk_id_offset: int = 0, max_retries: int = 3) -> Tuple[Dict, List[Dict]]:
        """
        Process a single Wikipedia URL and return metadata + chunks
        With retry logic for transient failures
        """
        for attempt in range(max_retries):
            try:
                title = url.split('/wiki/')[-1].replace('_', ' ')
                page = self.wiki.page(title)
                
                if not page.exists():
                    return None, []
                
                # Get text with timeout handling
                try:
                    page_text = page.text
                    page_summary = page.summary if hasattr(page, 'summary') else ""
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return None, []
                
                # Clean text
                clean_text = self.clean_text(page_text)
                
                # Chunk text
                chunks = self.chunk_text(
                    clean_text,
                    self.config['chunking']['min_tokens'],
                    self.config['chunking']['max_tokens'],
                    self.config['chunking']['overlap_tokens']
                )
                
                # Create metadata
                metadata = {
                    'url': url,
                    'title': page.title,
                    'word_count': len(page_text.split()),
                    'num_chunks': len(chunks),
                    'summary': page_summary[:500] if page_summary else ""
                }
                
                # Create chunk documents
                chunk_docs = []
                for i, chunk_text in enumerate(chunks):
                    chunk_docs.append({
                        'chunk_id': f"{chunk_id_offset + i}",
                        'url': url,
                        'title': page.title,
                        'text': chunk_text,
                        'token_count': self.count_tokens(chunk_text),
                        'chunk_index': i
                    })
                
                return metadata, chunk_docs
                
            except KeyboardInterrupt:
                raise  # Don't catch keyboard interrupt
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Final attempt failed
                    return None, []
        
        return None, []
    
    def collect_and_process_corpus(self, use_existing_fixed: bool = True) -> Dict:
        """
        Main method to collect and process entire corpus
        """
        fixed_urls_path = Path(self.config['data']['fixed_urls_file'])
        
        # Load or generate fixed URLs
        if use_existing_fixed and fixed_urls_path.exists():
            print("Loading existing fixed URLs...")
            with open(fixed_urls_path, 'r') as f:
                fixed_urls_data = json.load(f)
                fixed_urls = fixed_urls_data['urls']
        else:
            print("Generating new fixed URLs...")
            fixed_urls = self.generate_diverse_fixed_urls(
                self.config['data']['fixed_url_count']
            )
            
            # Save fixed URLs
            fixed_urls_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fixed_urls_path, 'w') as f:
                json.dump({
                    'urls': fixed_urls,
                    'count': len(fixed_urls),
                    'generated_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
            print(f"Saved {len(fixed_urls)} fixed URLs to {fixed_urls_path}")
        
        # Collect random URLs
        random_urls = self.collect_random_urls(
            self.config['data']['random_url_count'],
            set(fixed_urls)
        )
        
        # Combine all URLs
        all_urls = fixed_urls + random_urls
        print(f"\nTotal URLs to process: {len(all_urls)}")
        
        # Process all URLs
        corpus = {
            'metadata': {
                'total_urls': len(all_urls),
                'fixed_urls': len(fixed_urls),
                'random_urls': len(random_urls),
                'created_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'documents': [],
            'chunks': []
        }
        
        chunk_id_offset = 0
        
        print("\nProcessing Wikipedia articles...")
        for url in tqdm(all_urls):
            metadata, chunks = self.process_url(url, chunk_id_offset)
            
            if metadata and chunks:
                corpus['documents'].append(metadata)
                corpus['chunks'].extend(chunks)
                chunk_id_offset += len(chunks)
        
        corpus['metadata']['total_chunks'] = len(corpus['chunks'])
        
        # Save corpus
        corpus_path = Path(self.config['data']['corpus_file'])
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(corpus_path, 'w') as f:
            json.dump(corpus, f, indent=2)
        
        print(f"\n✓ Corpus saved to {corpus_path}")
        print(f"  Total documents: {len(corpus['documents'])}")
        print(f"  Total chunks: {len(corpus['chunks'])}")
        
        return corpus


def main():
    """Main execution"""
    collector = WikipediaDataCollector()
    
    print("=" * 70)
    print("HYBRID RAG DATA COLLECTION")
    print("=" * 70)
    print(f"Target: 200 fixed URLs + 300 random URLs = 500 total articles")
    print(f"This will take approximately 15-25 minutes")
    print(f"The process includes automatic retries and error handling")
    print("=" * 70)
    
    # Collect and process corpus with full requirements
    corpus = collector.collect_and_process_corpus(use_existing_fixed=True)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Fixed URLs: {corpus['metadata']['fixed_urls']}")
    print(f"Random URLs: {corpus['metadata']['random_urls']}")
    print(f"Total URLs: {corpus['metadata']['total_urls']}")
    print(f"Total Chunks: {corpus['metadata']['total_chunks']}")
    print("="*60)


if __name__ == "__main__":
    main()
