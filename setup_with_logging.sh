#!/bin/bash

# Hybrid RAG System - Setup with Detailed Logging
# This script shows detailed progress at each step

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="setup_log_$(date +%Y%m%d_%H%M%S).txt"

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}========================================${NC}" | tee -a "$LOG_FILE"
}

echo ""
log_step "HYBRID RAG SYSTEM - SETUP WITH LOGGING"
log "Log file: $LOG_FILE"
log "Start time: $(date)"
echo ""

# Step 1: Check Python
log_step "STEP 1: Checking Python"
log "Checking Python version..."
python_version=$(python --version 2>&1)
log "Found: $python_version"

# Step 2: Virtual Environment
log_step "STEP 2: Setting up Virtual Environment"
if [ ! -d "venv" ]; then
    log "Creating virtual environment..."
    python -m venv venv
    log "✓ Virtual environment created"
else
    log "✓ Virtual environment already exists"
fi

log "Activating virtual environment..."
source venv/bin/activate
log "✓ Virtual environment activated"

# Step 3: Install Dependencies (with progress)
log_step "STEP 3: Installing Dependencies"
log "This will take 5-10 minutes depending on your internet speed"
log "Installing packages with verbose output..."
echo ""

pip install --upgrade pip | tee -a "$LOG_FILE"

log_info "Installing core packages..."
pip install torch transformers sentence-transformers -v 2>&1 | tee -a "$LOG_FILE"

log_info "Installing FAISS..."
pip install faiss-cpu -v 2>&1 | tee -a "$LOG_FILE"

log_info "Installing remaining packages..."
pip install -r requirements.txt -v 2>&1 | tee -a "$LOG_FILE"

log "✓ All dependencies installed"

# Step 4: Download NLTK Data
log_step "STEP 4: Downloading NLTK Data"
log "Downloading punkt and stopwords..."
python -c "import nltk; nltk.download('punkt', quiet=False); nltk.download('stopwords', quiet=False)" | tee -a "$LOG_FILE"
log "✓ NLTK data downloaded"

# Step 5: Generate Fixed URLs
log_step "STEP 5: Generating Fixed URLs"
log "Creating 200 Wikipedia URLs across 12 domains..."
python generate_fixed_urls.py 2>&1 | tee -a "$LOG_FILE"
log "✓ Fixed URLs generated"

# Step 6: Collect Wikipedia Data
log_step "STEP 6: Collecting Wikipedia Data"
log "This will take 30-60 minutes..."
log "Scraping 500 Wikipedia articles (200 fixed + 300 random)..."
python src/data_collection.py 2>&1 | tee -a "$LOG_FILE"
log "✓ Wikipedia data collected"

# Step 7: Build Indexes
log_step "STEP 7: Building Search Indexes"
log "Building FAISS and BM25 indexes (10-20 minutes)..."
python -c "
import sys
sys.path.insert(0, '.')
from src.rag_system import HybridRAGSystem
print('[INFO] Initializing RAG system...')
rag = HybridRAGSystem()
print('[INFO] Loading corpus...')
rag.load_corpus()
print('[INFO] Building dense index...')
rag.build_dense_index()
print('[INFO] Building sparse index...')
rag.build_sparse_index()
print('[INFO] ✓ Indexes built successfully')
" 2>&1 | tee -a "$LOG_FILE"
log "✓ Indexes built"

# Step 8: Generate Questions
log_step "STEP 8: Generating Evaluation Questions"
log "Creating 100 Q&A pairs (5-10 minutes)..."
python src/question_generation.py 2>&1 | tee -a "$LOG_FILE"
log "✓ Questions generated"

# Step 9: Run Evaluation
log_step "STEP 9: Running Full Evaluation"
log "This will take 30-60 minutes..."
log "Calculating MRR, BERTScore, NDCG..."
python evaluation/pipeline.py 2>&1 | tee -a "$LOG_FILE"
log "✓ Evaluation complete"

# Summary
log_step "SETUP COMPLETE!"
log "End time: $(date)"
log "Total duration: Check log file for details"
echo ""
log "✓ Data collected: data/corpus.json"
log "✓ Indexes built: models/"
log "✓ Questions generated: data/questions_100.json"
log "✓ Evaluation complete: reports/"
echo ""
log "Full log saved to: $LOG_FILE"
echo ""
log_info "To launch the UI, run:"
echo "  streamlit run app.py"
echo ""
