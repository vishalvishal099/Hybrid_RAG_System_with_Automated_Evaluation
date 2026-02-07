"""
Monitor evaluation progress
"""
import time
import os

print("📊 Monitoring evaluation progress...")
print("Press Ctrl+C to stop monitoring\n")

last_size = 0
start_time = time.time()

try:
    while True:
        # Check if results file exists and is growing
        if os.path.exists('evaluation_results_chromadb.csv'):
            current_size = os.path.getsize('evaluation_results_chromadb.csv')
            if current_size != last_size:
                elapsed = time.time() - start_time
                print(f"⏱️  {elapsed/60:.1f} min elapsed | Results file: {current_size/1024:.1f} KB")
                last_size = current_size
        
        time.sleep(10)  # Check every 10 seconds
        
except KeyboardInterrupt:
    print("\n✓ Monitoring stopped")
