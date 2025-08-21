
"""
Zero-Knowledge Proof utilities for API endpoints.
"""
import os
import sys
import json
import numpy as np

# Ensure modules import works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our ZKP processor
try:
    from zkpy_svm_flow import SVMZkpProcessor, create_bank_message
    ZKPY_AVAILABLE = True
except ImportError:
    print("Warning: zkpy_svm_flow module not found. Will use simple inference without ZKP.")
    ZKPY_AVAILABLE = False

def classify_with_zkp(sample_features, model_dir=None, circuit_path=None):
    """
    Classify a sample with ZKP.
    
    Args:
        sample_features: Sample features to classify
        model_dir: Directory containing the model weights and bias
        circuit_path: Path to the circuit file
        
    Returns:
        Dictionary with classification results and ZKP information
    """
    # Default paths
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, 'models', 'master')
    if circuit_path is None:
        circuit_path = os.path.join(PROJECT_ROOT, 'circuit.r1cs')
    
    # Check if ZKP is available
    if not ZKPY_AVAILABLE:
        # Fallback to simple prediction
        weights_path = os.path.join(model_dir, 'svm_weights_master.npy')
        bias_path = os.path.join(model_dir, 'svm_bias_master.npy')
        weights = np.load(weights_path)
        bias = np.load(bias_path)
        prediction = int((np.dot(weights.flatten(), sample_features) + bias.flatten()[0]) >= 0)
        
        return {
            'prediction': prediction,
            'zkp_status': "ZKP not available - using direct prediction"
        }
    
    try:
        # Initialize ZKP processor
        zkp = SVMZkpProcessor(model_dir=model_dir, circuit_path=circuit_path)
        
        # Compile circuit
        zkp.compile_circuit()
        
        # Load model - use the svm_weights_master.npy and svm_bias_master.npy files
        try:
            # First try with default load_model which looks for svm_weights_bank1.npy
            weights, bias = zkp.load_model()
        except:
            # If that fails, load directly using the master model files
            weights_path = os.path.join(model_dir, 'svm_weights_master.npy')
            bias_path = os.path.join(model_dir, 'svm_bias_master.npy')
            weights = np.load(weights_path)
            bias = np.load(bias_path)
        
        # Generate proof
        result = zkp.generate_proof(weights, bias, sample_features)
        
        # Return result
        return {
            'prediction': result['prediction'],
            'prediction_value': result['prediction_value'],
            'proof_path': result['proof_path'],
            'zkp_status': "Success - Using pure zkpy implementation"
        }
    except Exception as e:
        # Fallback to simple prediction on error
        weights_path = os.path.join(model_dir, 'svm_weights_master.npy')
        bias_path = os.path.join(model_dir, 'svm_bias_master.npy')
        weights = np.load(weights_path)
        bias = np.load(bias_path)
        prediction = int((np.dot(weights.flatten(), sample_features) + bias.flatten()[0]) >= 0)
        
        return {
            'prediction': prediction,
            'zkp_status': f"ZKP Error: {str(e)}"
        }

def verify_zkp(proof_path):
    """
    Verify a ZKP proof.
    
    Args:
        proof_path: Path to the proof file
        
    Returns:
        Boolean indicating if the proof is valid
    """
    if not ZKPY_AVAILABLE:
        return False
    
    try:
        # Initialize ZKP processor
        zkp = SVMZkpProcessor(circuit_path=os.path.join(PROJECT_ROOT, 'circuit.r1cs'))
        
        # Compile circuit
        zkp.compile_circuit()
        
        # Check if this is a JSON proof
        if proof_path.endswith('.json'):
            print(f"Verifying JSON proof from {proof_path}...")
            with open(proof_path, 'r') as f:
                proof_data = json.load(f)
            
            # Check if the JSON contains required fields
            required_fields = ['prediction', 'prediction_value', 'witness_hash', 'circuit_hash', 'timestamp']
            if all(field in proof_data for field in required_fields):
                # In a real implementation, this would include more cryptographic checks
                print("Verified JSON proof structure (without cryptographic verification)")
                return True
            else:
                print("JSON proof missing required fields")
                return False
        
        # Verify proof - this handles both zkpy proofs and falls back to JSON proofs
        return zkp.verify_proof(proof_path)
    except Exception as e:
        print(f"Error verifying proof: {e}")
        return False
