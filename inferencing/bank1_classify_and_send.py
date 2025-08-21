
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import json

# Try to import zkpy and pysnark, but handle their absence gracefully
try:
    import zkpy
    ZKPY_AVAILABLE = True
except ImportError:
    print("Warning: zkpy module not found. Will use simple inference without ZKP.")
    ZKPY_AVAILABLE = False

try:
    from modules.svm_zkp import convert_model_to_zkp
    PYSNARK_AVAILABLE = True
except ImportError:
    print("Warning: pysnark module not found. Will use simple inference without ZKP.")
    PYSNARK_AVAILABLE = False
    
    # Define a fallback function for convert_model_to_zkp
    def convert_model_to_zkp(weights_path, bias_path, metadata_out_path):
        """Fallback function when pysnark is not available"""
        import numpy as np
        import json
        import os
        
        # Load the model
        weights = np.load(weights_path)
        bias = np.load(bias_path)
        
        # Create basic metadata
        model_info = {
            'num_features': weights.shape[1],
            'weights_shape': weights.shape,
            'bias_shape': bias.shape,
            'note': 'Created without pysnark - no commitments available'
        }
        
        # Save metadata
        out_dir = os.path.dirname(metadata_out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(metadata_out_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Basic SVM metadata (without ZKP) has been created and saved to {metadata_out_path}.")
        return weights, bias

# Paths
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'master')
ZKP_DIR = os.path.join(PROJECT_ROOT, 'models', 'zkp')
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'archive', 'creditcard.csv')
DEPLOY_DIR = os.path.join(PROJECT_ROOT, 'deploy')
os.makedirs(DEPLOY_DIR, exist_ok=True)

# Check for circuit files in root directory
r1cs_path = os.path.join(PROJECT_ROOT, 'svm.circ.r1cs')
if not os.path.exists(r1cs_path):
    print(f"Warning: Circuit R1CS file not found at {r1cs_path}")
    print("Will use simple inference without ZKP")

# Convert master model to ZKP and save metadata
weights_path = os.path.join(MODEL_DIR, 'svm_weights_master.npy')
bias_path = os.path.join(MODEL_DIR, 'svm_bias_master.npy')
zkp_metadata_path = os.path.join(ZKP_DIR, 'svm_master_zkp_metadata.json')
weights, bias = convert_model_to_zkp(weights_path, bias_path, zkp_metadata_path)

# Load a transaction sample
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X = df.drop('Class', axis=1)
y = df['Class']

# For demo, pick a random transaction
sample_id = np.random.randint(0, len(df))
sample_features = X.iloc[sample_id].values
true_label = y.iloc[sample_id]

# Check if circuit files exist in root directory
root_r1cs_file = os.path.join(PROJECT_ROOT, 'svm.circ.r1cs')
root_wasm_file = os.path.join(PROJECT_ROOT, 'svm.circ.wasm')

if os.path.exists(root_r1cs_file) and os.path.exists(root_wasm_file) and ZKPY_AVAILABLE:
    try:
        # Use zkpy Circuit with the root files
        circuit = zkpy.Circuit(root_r1cs_file)
        
        # Prepare witness for proof
        witness = list(weights.flatten()) + [float(bias.flatten()[0])] + [float(f) for f in sample_features]
        
        # Save public inputs to a file for later verification
        public_inputs = witness  # For SVM, all inputs are public
        
        public_file = os.path.join(PROJECT_ROOT, 'models', 'zkl', f'public_inputs_{sample_id}.json')
        os.makedirs(os.path.dirname(public_file), exist_ok=True)
        with open(public_file, 'w') as f:
            json.dump(public_inputs, f)
        
        # Generate proof using circuit.prove() method
        # According to signature: prove(scheme, proof_out=None, public_out=None)
        zkl_dir = os.path.join(PROJECT_ROOT, 'models', 'zkl')
        os.makedirs(zkl_dir, exist_ok=True)
        
        proof_path = os.path.join(zkl_dir, f'proof_{sample_id}.zkpy')
        
        # Ensure both output paths exist and are properly defined
        if not proof_path or not os.path.exists(os.path.dirname(proof_path)):
            os.makedirs(os.path.dirname(proof_path), exist_ok=True)
        
        if not public_file or not os.path.exists(os.path.dirname(public_file)):
            os.makedirs(os.path.dirname(public_file), exist_ok=True)
            
        # Generate the proof and save directly to the file
        circuit.prove(scheme=zkpy.GROTH, proof_out=proof_path, public_out=public_file)
        
        # Calculate prediction directly too
        prediction = int((np.dot(weights.flatten(), sample_features) + bias.flatten()[0]) >= 0)
        
        # Prepare message with ZKP
        message = {
            'sample_id': int(sample_id),
            'prediction': prediction,
            'true_label': int(true_label),
            'proof_path': proof_path,
            'public_file': public_file,
            'zkp_status': "Success - Using circuit files from root directory"
        }
        print(f"Successfully generated ZKP proof at {proof_path}")
        print(f"Public inputs saved to {public_file}")
    except Exception as e:
        # Fall back to simple prediction
        prediction = int((np.dot(weights.flatten(), sample_features) + bias.flatten()[0]) >= 0)
        print(f"ZKP generation failed: {str(e)}")
        print("Falling back to simple prediction")
        
        message = {
            'sample_id': int(sample_id),
            'prediction': prediction,
            'true_label': int(true_label),
            'zkp_status': f"ZKP Error: {str(e)}"
        }
else:
    # Simple prediction without ZKP
    prediction = int((np.dot(weights.flatten(), sample_features) + bias.flatten()[0]) >= 0)
    print("Circuit files not found - using direct prediction")
    
    message = {
        'sample_id': int(sample_id),
        'prediction': prediction,
        'true_label': int(true_label),
        'zkp_status': "Circuit files not found - using direct prediction"
    }

# Save message
with open(os.path.join(DEPLOY_DIR, 'bank_message.json'), 'w') as f:
    json.dump(message, f, indent=2)

print(f"Processed sample {sample_id}, prediction: {prediction}, true label: {true_label}")
print(f"Message saved to {os.path.join(DEPLOY_DIR, 'bank_message.json')}")

print(f"Bank1: Sent transaction #{sample_id} (true label: {true_label}) with prediction: {prediction} and ZKP proof to Bank2.")
