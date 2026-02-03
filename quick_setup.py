"""
Quick Setup Script - Creates a working corpus quickly for testing
Uses a smaller subset of Wikipedia articles with better error handling
"""

import json
import time
from pathlib import Path
import wikipediaapi
from tqdm import tqdm
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='HybridRAG/1.0 (Educational Project; Contact: student@example.com)'
)

# Curated list of reliable Wikipedia articles (these are stable and well-structured)
RELIABLE_ARTICLES = [
    # Science & Technology
    "Artificial_intelligence", "Machine_learning", "Deep_learning", "Natural_language_processing",
    "Computer_vision", "Robotics", "Quantum_computing", "Blockchain", "Cryptocurrency",
    "Python_(programming_language)", "JavaScript", "Java_(programming_language)",
    
    # Physics & Chemistry
    "Physics", "Quantum_mechanics", "General_relativity", "Thermodynamics", "Electromagnetism",
    "Chemistry", "Organic_chemistry", "Periodic_table", "Chemical_bond", "Photosynthesis",
    
    # Biology & Medicine
    "Biology", "Evolution", "DNA", "Cell_(biology)", "Genetics", "Ecology", 
    "Human_body", "Brain", "Heart", "Immune_system",
    
    # Mathematics
    "Mathematics", "Calculus", "Linear_algebra", "Statistics", "Probability",
    "Number_theory", "Graph_theory", "Topology",
    
    # History
    "History", "Ancient_Egypt", "Roman_Empire", "Middle_Ages", "Renaissance",
    "Industrial_Revolution", "World_War_I", "World_War_II", "Cold_War",
    
    # Geography
    "Geography", "Earth", "Atmosphere_of_Earth", "Ocean", "Mountain",
    "Climate_change", "Global_warming", "Renewable_energy",
    
    # Philosophy & Social Sciences
    "Philosophy", "Ethics", "Logic", "Metaphysics", "Epistemology",
    "Psychology", "Sociology", "Economics", "Political_science",
    
    # Arts & Culture
    "Art", "Music", "Literature", "Film", "Theatre", "Architecture",
    "Leonardo_da_Vinci", "William_Shakespeare", "Ludwig_van_Beethoven",
    
    # Sports
    "Association_football", "Basketball", "Cricket", "Tennis", "Olympic_Games",
    
    # Additional diverse topics
    "Language", "Religion", "Education", "Technology", "Internet",
    "Social_media", "Climate", "Biodiversity", "Sustainable_development",
    "Human_rights", "Democracy", "Capitalism", "Socialism",
    "Ancient_Greece", "Ancient_Rome", "Ancient_China", "Ancient_India",
    "Medieval_Europe", "Age_of_Discovery", "Colonialism", "Decolonization",
    "Space_exploration", "Astronomy", "Cosmology", "Big_Bang",
    "Black_hole", "Star", "Galaxy", "Solar_System",
    "Atom", "Molecule", "Element_(chemistry)", "Compound_(chemistry)",
    "Protein", "Enzyme", "Virus", "Bacteria",
    "Neuroscience", "Consciousness", "Memory", "Learning",
    "Artificial_neural_network", "Convolutional_neural_network",
    "Recurrent_neural_network", "Transformer_(machine_learning)",
    "Data_science", "Big_data", "Cloud_computing", "Edge_computing",
    "Internet_of_things", "Cybersecurity", "Encryption", "Privacy",
    "Sustainable_energy", "Solar_power", "Wind_power", "Nuclear_power",
    "Electric_vehicle", "Autonomous_car", "Smart_city"
]

print(f"Quick Setup: Using {len(RELIABLE_ARTICLES)} curated Wikipedia articles")
print("=" * 60)

corpus = {
    'metadata': {
        'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Wikipedia (Curated Articles)',
        'total_urls': 0,
        'fixed_urls': len(RELIABLE_ARTICLES),
        'random_urls': 0,
        'total_chunks': 0
    },
    'documents': [],
    'chunks': []
}

successful = 0
failed = 0
chunk_id = 0

print("\nFetching articles...")
for article in tqdm(RELIABLE_ARTICLES):
    try:
        page = wiki.page(article)
        
        if not page.exists():
            failed += 1
            continue
            
        text = page.text
        if len(text.split()) < config['data']['min_words_per_page']:
            failed += 1
            continue
        
        # Clean text
        import re
        text = re.sub(r'\[\d+\]', '', text)  # Remove citations
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        
        # Simple chunking by paragraphs (for speed)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        # Create document entry
        doc_id = f"doc_{successful}"
        corpus['documents'].append({
            'doc_id': doc_id,
            'url': page.fullurl,
            'title': page.title,
            'word_count': len(text.split()),
            'num_chunks': len(paragraphs)
        })
        
        # Create chunks
        for i, para in enumerate(paragraphs[:10]):  # Max 10 chunks per article
            corpus['chunks'].append({
                'chunk_id': f"chunk_{chunk_id}",
                'doc_id': doc_id,
                'text': para,
                'doc_title': page.title,
                'url': page.fullurl
            })
            chunk_id += 1
        
        successful += 1
        time.sleep(0.1)  # Rate limiting
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        break
    except Exception as e:
        failed += 1
        continue

# Update metadata
corpus['metadata']['total_urls'] = successful
corpus['metadata']['total_chunks'] = len(corpus['chunks'])

# Save corpus
output_path = Path(config['data']['corpus_file'])
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(corpus, f, indent=2)

print("\n" + "=" * 60)
print("QUICK SETUP COMPLETE!")
print("=" * 60)
print(f"✓ Articles collected: {successful}")
print(f"✗ Articles failed: {failed}")
print(f"✓ Total chunks created: {len(corpus['chunks'])}")
print(f"✓ Corpus saved to: {output_path}")
print("=" * 60)
print("\nNext steps:")
print("1. Build indexes: python src/indexing.py")
print("2. Generate questions: python src/evaluation/question_generation.py")
print("3. Run evaluation: python src/evaluation/evaluation_pipeline.py")
print("4. Launch UI: streamlit run app.py")
print("=" * 60)
