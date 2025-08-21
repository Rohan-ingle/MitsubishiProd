"""
Bank1: Classify transactions and send results with ZKP to Bank2 using pure zkpy
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import json
import random

# Import our ZKP processor
try:
    from zkpy_svm_flow import SVMZkpProcessor, create_bank_message
    ZKPY_AVAILABLE = True
except ImportError:
    print("Error: zkpy_svm_flow.py not found. Make sure you've created this file.")
    ZKPY_AVAILABLE = False

# Set paths
BANK1_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'bank1')
DEPLOY_DIR = os.path.join(PROJECT_ROOT, 'deploy')
os.makedirs(DEPLOY_DIR, exist_ok=True)

def load_sample_data(sample_id=None):
    """
    Load a transaction sample from the dataset.
    
    Args:
        sample_id: ID of the sample to load (if None, a random one is selected)
        
    Returns:
        Tuple of (features, true_label, sample_id)
    """
    try:
        # Try to load from creditcard.csv
        csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'archive', 'creditcard.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Select a random sample if no sample_id is provided
            if sample_id is None:
                sample_id = random.randint(0, len(df) - 1)
            
            # Extract features and label
            row = df.iloc[sample_id]
            features = row[df.columns[:-1]].values  # All columns except the last one
            label = int(row['Class'])
            
            return features, label, sample_id
    except Exception as e:
        print(f"Error loading sample data: {e}")
    
    # Fallback: Generate random data
    print("Using random sample data")
    features = np.random.rand(30)  # Assuming 30 features
    label = random.choice([0, 1])
    if sample_id is None:
        sample_id = 1
    
    return features, label, sample_id

def main(sample_id=None):
    """
    Main function to classify a transaction and send the result with ZKP.
    
    Args:
        sample_id: ID of the sample to classify (if None, a random one is selected)
    """
    if not ZKPY_AVAILABLE:
        print("ZKP functionality not available. Please check zkpy_svm_flow.py.")
        return
    
    # Load sample data
    features, true_label, sample_id = load_sample_data(sample_id)
    print(f"Processing transaction #{sample_id}")
    
    # Initialize ZKP processor
    zkp = SVMZkpProcessor(model_dir=BANK1_MODEL_DIR)
    
    # Compile circuit
    try:
        zkp.compile_circuit()
    except Exception as e:
        print(f"Error compiling circuit: {e}")
        return
    
    # Load model
    try:
        weights, bias = zkp.load_model()
        print(f"Loaded model: weights shape {weights.shape}, bias {bias}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate proof
    try:
        result = zkp.generate_proof(weights, bias, features)
        prediction = result['prediction']
        proof_path = result['proof_path']
    except Exception as e:
        print(f"Error generating proof: {e}")
        return
    
    # Create bank message
    try:
        message = create_bank_message(sample_id, prediction, proof_path, true_label)
    except Exception as e:
        print(f"Error creating bank message: {e}")
        return
    
    # Print summary
    print("\nSummary:")
    print(f"Transaction #{sample_id}")
    print(f"Prediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
    print(f"True Label: {true_label}")
    print(f"Proof generated: {os.path.basename(proof_path)}")
    print(f"Message sent to Bank2 at {os.path.join(DEPLOY_DIR, 'bank_message.json')}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Bank1: Classify transactions with ZKP')
    parser.add_argument('--sample-id', type=int, help='ID of the sample to classify')
    args = parser.parse_args()
    
    main(args.sample_id)
