"""
Bank2: Receive transaction results from Bank1 and verify the ZKP proof using pure zkpy
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

# Import our ZKP processor
try:
    from zkpy_svm_flow import SVMZkpProcessor
    ZKPY_AVAILABLE = True
except ImportError:
    print("Error: zkpy_svm_flow.py not found. Make sure you've created this file.")
    ZKPY_AVAILABLE = False

# Set paths
DEPLOY_DIR = os.path.join(PROJECT_ROOT, 'deploy')
MESSAGE_PATH = os.path.join(DEPLOY_DIR, 'bank_message.json')

def main():
    """Main function to receive and verify transaction results."""
    if not ZKPY_AVAILABLE:
        print("ZKP functionality not available. Please check zkpy_svm_flow.py.")
        return
    
    # Check if message file exists
    if not os.path.exists(MESSAGE_PATH):
        print(f"Error: Message file not found at {MESSAGE_PATH}")
        return
    
    # Load message
    try:
        with open(MESSAGE_PATH, 'r') as f:
            message = json.load(f)
        
        sample_id = message['sample_id']
        prediction = message['prediction']
        proof_path = message['proof_path']
        y_true = message.get('y_value', -1)  # -1 indicates unknown
    except Exception as e:
        print(f"Error loading message: {e}")
        return
    
    print(f"Bank 2 received transaction #{sample_id}.")
    
    # Initialize ZKP processor
    zkp = SVMZkpProcessor()
    
    # Compile circuit (needed for verification)
    try:
        zkp.compile_circuit()
    except Exception as e:
        print(f"Error compiling circuit: {e}")
        return
    
    # Verify proof
    try:
        # Check if proof file exists
        if not os.path.exists(proof_path):
            print(f"Error: Proof file not found at {proof_path}")
            verified = False
        else:
            verified = zkp.verify_proof(proof_path)
    except Exception as e:
        print(f"Error verifying proof: {e}")
        verified = False
    
    # Print results
    print(f"\nBank 2 received and verified transaction #{sample_id}.")
    print(f"\nPrediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
    print(f"\nZKP Verified: {verified}")
    
    if y_true != -1:
        print(f"\nTrue Label: {y_true}")
    
    # Take action based on verification
    if verified:
        print("\nVerification successful: The classification result is trustworthy.")
        if prediction == 1:
            print("Action: Transaction flagged for fraud investigation.")
        else:
            print("Action: Transaction processed normally.")
    else:
        print("\nVerification failed: Cannot trust the classification result.")
        print("Action: Rejecting the result and requesting re-submission.")

if __name__ == "__main__":
    main()
