"""
Helper script to run the complete setup
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    print("="*60)
    print("HYBRID RAG SYSTEM - COMPLETE SETUP")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("Error: config.yaml not found. Please run from project root.")
        sys.exit(1)
    
    # Step 1: Data Collection
    if not Path("data/corpus.json").exists():
        print("\n🔹 Step 1: Data Collection")
        print("This will collect 500 Wikipedia articles (200 fixed + 300 random)")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            run_command("python src/data_collection.py", "Data Collection")
    else:
        print("\n✓ Corpus already exists, skipping data collection")
    
    # Step 2: Build Indexes
    if not Path("models/faiss_index").exists():
        print("\n🔹 Step 2: Build Indexes")
        print("This will build FAISS and BM25 indexes")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            cmd = """python -c "from src.rag_system import HybridRAGSystem; rag = HybridRAGSystem(); rag.load_corpus(); rag.build_dense_index(); rag.build_sparse_index()" """
            run_command(cmd, "Index Building")
    else:
        print("\n✓ Indexes already exist, skipping index building")
    
    # Step 3: Generate Questions
    if not Path("data/questions_100.json").exists():
        print("\n🔹 Step 3: Generate Questions")
        print("This will generate 100 evaluation questions")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            run_command("python src/question_generation.py", "Question Generation")
    else:
        print("\n✓ Questions already exist, skipping question generation")
    
    # Step 4: Run Evaluation
    print("\n🔹 Step 4: Run Evaluation")
    print("This will run the complete evaluation pipeline")
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        run_command("python evaluation/pipeline.py", "Evaluation Pipeline")
    
    # Step 5: Launch UI
    print("\n🔹 Step 5: Launch Streamlit UI")
    response = input("Launch UI now? (y/n): ")
    if response.lower() == 'y':
        print("\nLaunching Streamlit UI...")
        print("Access at: http://localhost:8501")
        print("Press Ctrl+C to stop")
        run_command("streamlit run app.py", "Streamlit UI")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nTo launch the UI later, run:")
    print("  streamlit run app.py")
    print("\nTo re-run evaluation:")
    print("  python evaluation/pipeline.py")


if __name__ == "__main__":
    main()
