"""
Helper script to run tests and start the API server
"""
import subprocess
import os
import sys
import time
from multiprocessing import Process

def run_api_server():
    """Start the API server in a separate process"""
    print("Starting the API server...")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the FastAPI server
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"])

def run_tests():
    """Run the test scripts"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n===== Testing Market Agent =====")
    subprocess.run([sys.executable, "test_market_agent.py"])
    
    print("\n===== Testing Real Web Scraping =====")
    subprocess.run([sys.executable, "test_real_scraping.py"])
    
    print("\n===== Testing API with Real Scraping =====")
    subprocess.run([sys.executable, "test_api_real_scraping.py"])

def main():
    """Main function to start tests and API server"""
    print("=== AI Luxury Brand Market Agent Test Suite ===\n")
    
    # Start API server in a separate process
    api_process = Process(target=run_api_server)
    api_process.start()
    
    # Wait for API server to start
    print("Waiting for API server to start...")
    time.sleep(5)
    
    try:
        # Run the tests
        run_tests()
    finally:
        # Clean up
        print("\nShutting down API server...")
        api_process.terminate()
        api_process.join(timeout=5)
        print("Done!")

if __name__ == "__main__":
    main()
