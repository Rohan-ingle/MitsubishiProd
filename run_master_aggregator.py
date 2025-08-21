"""
Master Aggregator Script

This script aggregates the Bank1 and Bank2 models into a master model,
generates ZKP metadata, and tests the aggregated model with ZKP.

Run this script directly to aggregate models without using the API.
"""
import os
import sys
import numpy as np
import json

# Ensure modules import works
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.federated_utils import aggregate_master_model
from modules.svm_zkp import convert_model_to_zkp

def run_master_aggregator():
    """
    Aggregate Bank1 and Bank2 models into a master model and test the result.
    """
    print("Starting Master Aggregator...")
    
    # Define directories
    bank1_dir = os.path.join(PROJECT_ROOT, 'models', 'bank1')
    bank2_dir = os.path.join(PROJECT_ROOT, 'models', 'bank2')
    master_dir = os.path.join(PROJECT_ROOT, 'models', 'master')
    zkp_dir = os.path.join(PROJECT_ROOT, 'models', 'zkp')
    
    # Create directories if they don't exist
    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(zkp_dir, exist_ok=True)
    
    # Check if required files exist
    required_files = [
        os.path.join(bank1_dir, 'svm_weights_bank1.npy'),
        os.path.join(bank1_dir, 'svm_bias_bank1.npy'),
        os.path.join(bank2_dir, 'svm_weights_bank2.npy'),
        os.path.join(bank2_dir, 'svm_bias_bank2.npy')
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            return False
    
    # Aggregate models
    print(f"Aggregating models from {bank1_dir} and {bank2_dir}...")
    
    # Get the path to the creditcard.csv dataset
    data_path = os.path.join(PROJECT_ROOT, 'dataset', 'archive', 'creditcard.csv')
    
    # Pass optimize_bias=True, the data_path, and the metric to optimize for to the aggregator
    # Using 'gmean' (geometric mean of recall and specificity) is better for severely imbalanced data
    print("Using geometric mean of recall and specificity for bias optimization - best for imbalanced datasets")
    success = aggregate_master_model(bank1_dir, bank2_dir, master_dir, optimize_bias=True, 
                                   data_path=data_path, target_metric='gmean')
    
    if not success:
        print("Error: Model aggregation failed")
        return False
    
    print("Model aggregation completed successfully.")
    
    # Generate ZKP metadata
    weights_path = os.path.join(master_dir, 'svm_weights_master.npy')
    bias_path = os.path.join(master_dir, 'svm_bias_master.npy')
    zkp_metadata_path = os.path.join(zkp_dir, 'svm_master_zkp_metadata.json')
    
    print(f"Generating ZKP metadata for master model...")
    weights, bias = convert_model_to_zkp(weights_path, bias_path, zkp_metadata_path)
    
    # Verify that the aggregate model was created
    if not os.path.exists(weights_path) or not os.path.exists(bias_path):
        print("Error: Failed to create aggregated model files")
        return False
    
    print(f"Aggregated model files created:")
    print(f"  Weights: {weights_path}")
    print(f"  Bias: {bias_path}")
    print(f"  ZKP Metadata: {zkp_metadata_path}")
    
    # Test the aggregated model with ZKP
    try:
        # Import our ZKP processor
        from zkpy_svm_flow import SVMZkpProcessor
        
        # Generate a random sample for testing
        sample_features = np.random.rand(weights.shape[1])
        
        # Initialize ZKP processor with master model
        circuit_path = os.path.join(PROJECT_ROOT, 'circuit.r1cs')
        print(f"Testing ZKP with circuit: {circuit_path}")
        
        zkp = SVMZkpProcessor(model_dir=master_dir, circuit_path=circuit_path)
        
        # Compile circuit
        zkp.compile_circuit()
        
        # Generate a test proof
        print(f"Testing ZKP with the aggregated model...")
        result = zkp.generate_proof(weights, bias, sample_features)
        
        if result and 'proof_path' in result and os.path.exists(result['proof_path']):
            # Test verification
            verified = zkp.verify_proof(result['proof_path'])
            
            if verified:
                print(f"ZKP test successful: Generated and verified proof with aggregated model")
                print(f"Proof saved to: {result['proof_path']}")
            else:
                print(f"Warning: ZKP test proof was generated but verification failed")
        else:
            print(f"Warning: ZKP test failed to generate a proof")
    except Exception as e:
        print(f"Warning: ZKP test encountered an error: {str(e)}")
    
    print("\nMaster Aggregator completed successfully!")
    return True

if __name__ == "__main__":
    run_master_aggregator()
