@echo off
REM Hybrid RAG System - Complete Setup and Execution Script (Windows)
REM This script runs all necessary steps to set up and evaluate the system

echo ==========================================
echo HYBRID RAG SYSTEM - AUTOMATED SETUP
echo ==========================================
echo.

REM Check Python version
echo [INFO] Checking Python version...
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Install dependencies
echo [INFO] Installing dependencies...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
echo [OK] Dependencies installed

REM Download NLTK data
echo [INFO] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
echo [OK] NLTK data downloaded

echo.
echo ==========================================
echo STEP 1: DATA COLLECTION
echo ==========================================
echo.

REM Generate fixed URLs if not exists
if not exist "data\fixed_urls.json" (
    echo [INFO] Generating fixed URLs...
    python generate_fixed_urls.py
    echo [OK] Fixed URLs generated
) else (
    echo [OK] Fixed URLs already exist
)

REM Collect Wikipedia data
if not exist "data\corpus.json" (
    echo [INFO] Collecting Wikipedia data (this may take 30-60 minutes)...
    python src\data_collection.py
    echo [OK] Data collection complete
) else (
    echo [OK] Corpus already exists
)

echo.
echo ==========================================
echo STEP 2: INDEX BUILDING
echo ==========================================
echo.

REM Build indexes
if not exist "models\faiss_index" (
    echo [INFO] Building FAISS and BM25 indexes (this may take 10-20 minutes)...
    python -c "from src.rag_system import HybridRAGSystem; rag = HybridRAGSystem(); rag.load_corpus(); rag.build_dense_index(); rag.build_sparse_index()"
    echo [OK] Indexes built successfully
) else (
    echo [OK] Indexes already exist
)

echo.
echo ==========================================
echo STEP 3: QUESTION GENERATION
echo ==========================================
echo.

REM Generate questions
if not exist "data\questions_100.json" (
    echo [INFO] Generating 100 evaluation questions (this may take 5-10 minutes)...
    python src\question_generation.py
    echo [OK] Questions generated
) else (
    echo [OK] Questions already exist
)

echo.
echo ==========================================
echo STEP 4: EVALUATION
echo ==========================================
echo.

REM Run evaluation
echo [INFO] Running full evaluation pipeline (this may take 30-60 minutes)...
python evaluation\pipeline.py
echo [OK] Evaluation complete

echo.
echo ==========================================
echo SETUP COMPLETE!
echo ==========================================
echo.
echo Summary:
echo   - Data collected: 500 Wikipedia articles
echo   - Indexes built: FAISS + BM25
echo   - Questions generated: 100 diverse Q&A pairs
echo   - Evaluation complete: Results in reports/
echo.
echo Next steps:
echo   1. Review results:
echo      - reports\evaluation_results.json
echo      - reports\evaluation_results.csv
echo      - reports\visualizations\*.png
echo.
echo   2. Launch UI:
echo      streamlit run app.py
echo.
echo   3. Access at:
echo      http://localhost:8501
echo.
echo ==========================================

REM Ask if user wants to launch UI
set /p launch_ui="Launch Streamlit UI now? (y/n): "

if /i "%launch_ui%"=="y" (
    echo [INFO] Launching Streamlit UI...
    echo Access at: http://localhost:8501
    echo Press Ctrl+C to stop
    streamlit run app.py
)
