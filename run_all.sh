#!/bin/bash

# Hybrid RAG System - Complete Setup and Execution Script
# This script runs all necessary steps to set up and evaluate the system

set -e  # Exit on error

echo "=========================================="
echo "HYBRID RAG SYSTEM - AUTOMATED SETUP"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Install dependencies
print_info "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_status "Dependencies installed"

# Download NLTK data
print_info "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
print_status "NLTK data downloaded"

echo ""
echo "=========================================="
echo "STEP 1: DATA COLLECTION"
echo "=========================================="
echo ""

# Generate fixed URLs if not exists
if [ ! -f "data/fixed_urls.json" ]; then
    print_info "Generating fixed URLs..."
    python generate_fixed_urls.py
    print_status "Fixed URLs generated"
else
    print_status "Fixed URLs already exist"
fi

# Collect Wikipedia data
if [ ! -f "data/corpus.json" ]; then
    print_info "Collecting Wikipedia data (this may take 30-60 minutes)..."
    python src/data_collection.py
    print_status "Data collection complete"
else
    print_status "Corpus already exists"
fi

echo ""
echo "=========================================="
echo "STEP 2: INDEX BUILDING"
echo "=========================================="
echo ""

# Build indexes
if [ ! -f "models/faiss_index" ]; then
    print_info "Building FAISS and BM25 indexes (this may take 10-20 minutes)..."
    python -c "from src.rag_system import HybridRAGSystem; rag = HybridRAGSystem(); rag.load_corpus(); rag.build_dense_index(); rag.build_sparse_index()"
    print_status "Indexes built successfully"
else
    print_status "Indexes already exist"
fi

echo ""
echo "=========================================="
echo "STEP 3: QUESTION GENERATION"
echo "=========================================="
echo ""

# Generate questions
if [ ! -f "data/questions_100.json" ]; then
    print_info "Generating 100 evaluation questions (this may take 5-10 minutes)..."
    python src/question_generation.py
    print_status "Questions generated"
else
    print_status "Questions already exist"
fi

echo ""
echo "=========================================="
echo "STEP 4: EVALUATION"
echo "=========================================="
echo ""

# Run evaluation
print_info "Running full evaluation pipeline (this may take 30-60 minutes)..."
python evaluation/pipeline.py
print_status "Evaluation complete"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Data collected: 500 Wikipedia articles"
echo "  ✓ Indexes built: FAISS + BM25"
echo "  ✓ Questions generated: 100 diverse Q&A pairs"
echo "  ✓ Evaluation complete: Results in reports/"
echo ""
echo "Next steps:"
echo "  1. Review results:"
echo "     - reports/evaluation_results.json"
echo "     - reports/evaluation_results.csv"
echo "     - reports/visualizations/*.png"
echo ""
echo "  2. Launch UI:"
echo "     streamlit run app.py"
echo ""
echo "  3. Access at:"
echo "     http://localhost:8501"
echo ""
echo "=========================================="

# Ask if user wants to launch UI
read -p "Launch Streamlit UI now? (y/n): " launch_ui

if [ "$launch_ui" = "y" ] || [ "$launch_ui" = "Y" ]; then
    print_info "Launching Streamlit UI..."
    echo "Access at: http://localhost:8501"
    echo "Press Ctrl+C to stop"
    streamlit run app.py
fi
