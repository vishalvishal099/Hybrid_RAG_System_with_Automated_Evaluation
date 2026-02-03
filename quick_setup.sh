#!/bin/bash

# Quick Setup with Detailed Logging
# This version shows all progress and is faster for testing

set -e  # Exit on error

echo "========================================"
echo "QUICK SETUP - WITH DETAILED LOGGING"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Check Python
log "Checking Python version..."
python --version
success "Python version OK"
echo ""

# Step 2: Activate venv
log "Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    success "Virtual environment activated"
else
    log "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    success "Virtual environment created and activated"
fi
echo ""

# Step 3: Install dependencies with progress
log "Installing Python packages (this takes 5-10 minutes)..."
log "You'll see detailed progress for each package..."
echo ""

# Install packages one by one with progress
packages=(
    "numpy>=1.24.0"
    "pandas>=2.0.0"
    "pyyaml>=6.0"
    "tqdm>=4.65.0"
    "requests>=2.31.0"
    "beautifulsoup4>=4.12.0"
    "lxml>=4.9.0"
    "nltk>=3.8.0"
    "scikit-learn>=1.3.0"
    "matplotlib>=3.7.0"
    "seaborn>=0.12.0"
    "plotly>=5.14.0"
    "streamlit>=1.29.0"
    "wikipediaapi>=0.6.0"
    "rank-bm25>=0.2.2"
    "tiktoken>=0.5.1"
    "rouge-score>=0.1.2"
)

log "Installing basic packages first..."
for package in "${packages[@]}"; do
    log "Installing $package..."
    pip install "$package" -q
    success "$package installed"
done

log "Installing large packages (torch, transformers, faiss)..."
log "These are large (~2GB total) and will take longer..."
echo ""

log "Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --progress-bar on
success "PyTorch installed"
echo ""

log "Installing Transformers..."
pip install transformers -q
success "Transformers installed"
echo ""

log "Installing Sentence Transformers..."
pip install sentence-transformers -q
success "Sentence Transformers installed"
echo ""

log "Installing FAISS..."
pip install faiss-cpu -q
success "FAISS installed"
echo ""

log "Installing BERTScore..."
pip install bert-score -q
success "BERTScore installed"
echo ""

success "All dependencies installed!"
echo ""

# Step 4: Download NLTK data
log "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
success "NLTK data downloaded"
echo ""

# Step 5: Verify installation
log "Verifying installation..."
python -c "
import torch
import transformers
import sentence_transformers
import faiss
import streamlit
print('✓ All major packages imported successfully')
"
success "Installation verified!"
echo ""

echo "========================================"
echo "INSTALLATION COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Generate URLs:        python generate_fixed_urls.py"
echo "  2. Collect data:         python src/data_collection.py"
echo "  3. Test system:          python quick_test.py"
echo "  4. Launch UI:            streamlit run app.py"
echo ""
