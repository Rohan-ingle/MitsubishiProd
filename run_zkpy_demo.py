"""
Run the complete ZKP flow using pure zkpy implementation.

This script demonstrates the end-to-end flow of:
1. Bank1 classifying a transaction and generating a ZKP proof
2. Bank2 receiving the result and verifying the ZKP proof

Usage:
    python run_zkpy_demo.py [--sample-id SAMPLE_ID]
"""
import os
import sys
import subprocess
import time

# Ensure proper path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_command(cmd, description):
    """Run a command and print the output"""
    print(f"\n--- {description} ---")
    print(f"Running: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run the complete ZKP flow"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run ZKP Demo with pure zkpy')
    parser.add_argument('--sample-id', type=int, help='ID of the sample to classify')
    args = parser.parse_args()
    
    # Build command for Bank1
    bank1_cmd = [sys.executable, os.path.join(PROJECT_ROOT, 'inferencing', 'bank1_classify_and_send_zkpy.py')]
    if args.sample_id is not None:
        bank1_cmd.extend(['--sample-id', str(args.sample_id)])
    
    # Run Bank1 script to classify transaction and generate proof
    if not run_command(bank1_cmd, "Bank1: Classify transaction and generate ZKP proof"):
        print("Bank1 script failed. Aborting.")
        return
    
    # Add a small delay to ensure files are properly saved
    time.sleep(1)
    
    # Run Bank2 script to verify the proof
    bank2_cmd = [sys.executable, os.path.join(PROJECT_ROOT, 'inferencing', 'bank2_receive_and_verify_zkpy.py')]
    if not run_command(bank2_cmd, "Bank2: Receive and verify ZKP proof"):
        print("Bank2 script failed.")
    
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()
