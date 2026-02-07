#!/bin/bash

# Start UI Script for ChromaDB Hybrid RAG
# This script ensures proper startup with progress feedback

cd "$(dirname "$0")"

echo "======================================"
echo "  ChromaDB Hybrid RAG - Starting UI"
echo "======================================"
echo ""

# Activate virtual environment
echo "⚙️  Activating virtual environment..."
source venv/bin/activate

# Kill any existing Streamlit processes
echo "🧹 Cleaning up existing processes..."
pkill -9 -f "streamlit.*app_chromadb" 2>/dev/null
sleep 2

# Check if ChromaDB exists
if [ ! -f "chroma_db/chroma.sqlite3" ]; then
    echo "❌ ERROR: ChromaDB not found!"
    echo "   Please run: python build_chromadb_system.py"
    exit 1
fi

echo "✓ ChromaDB found ($(du -sh chroma_db/chroma.sqlite3 | cut -f1))"
echo ""

# Start Streamlit in background
echo "🚀 Starting Streamlit UI..."
echo "   This may take 30-60 seconds to load ChromaDB and models..."
echo ""

nohup python -m streamlit run app_chromadb.py \
    --server.port=8501 \
    --server.headless=true \
    --server.runOnSave=true \
    --server.maxUploadSize=200 \
    > streamlit_startup.log 2>&1 &

STREAMLIT_PID=$!

echo "   Process ID: $STREAMLIT_PID"
echo ""

# Wait for server to start
echo "⏳ Waiting for server to start..."
for i in {1..30}; do
    sleep 2
    if lsof -i :8501 >/dev/null 2>&1; then
        echo ""
        echo "✅ SERVER IS READY!"
        echo ""
        echo "======================================"
        echo "  Access the UI at:"
        echo "  🌐 http://localhost:8501"
        echo "======================================"
        echo ""
        echo "📝 Logs: tail -f streamlit_startup.log"
        echo "🛑 Stop: pkill -f streamlit"
        echo ""
        exit 0
    fi
    echo -n "."
done

echo ""
echo "⚠️  Server taking longer than expected..."
echo "📋 Check logs: tail -f streamlit_startup.log"
echo ""
