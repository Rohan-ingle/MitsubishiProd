import numpy as np
import json
import os

# Try to import pysnark, but handle its absence gracefully
try:
    from pysnark.runtime import PrivVal, snark
    PYSNARK_AVAILABLE = True
except ImportError:
    print("Warning: pysnark module not found. Will use simple SVM model without ZKP.")
    PYSNARK_AVAILABLE = False
    
    # Define a fallback decorator
    def snark(func):
        """Fallback decorator when pysnark is not available"""
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    # Define a fallback PrivVal class
    class PrivVal:
        """Fallback PrivVal class when pysnark is not available"""
        def __init__(self, value):
            self.value = value
        
        def __mul__(self, other):
            if isinstance(other, PrivVal):
                return PrivVal(self.value * other.value)
            return PrivVal(self.value * other)
        
        def __add__(self, other):
            if isinstance(other, PrivVal):
                return PrivVal(self.value + other.value)
            return PrivVal(self.value + other)

def commit_metadata(weights, bias, out_path):
    """Create and save model metadata with commitments (for reference)."""
    num_features = weights.shape[1]
    model_info = {
        'num_features': num_features,
        'weights_shape': weights.shape,
        'bias_shape': bias.shape,
        'weights_commitment': str(hash(tuple(weights.flatten()))),
        'bias_commitment': str(hash(tuple(bias.flatten())))
    }
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"SVM ZKP metadata has been created and saved to {out_path}.")

def load_model(weights_path, bias_path):
    weights = np.load(weights_path)
    bias = np.load(bias_path)
    return weights, bias

@snark
def svm_predict_zkp(w, b, x):
    # All arguments are lists of PrivVal
    dot = sum([wi * xi for wi, xi in zip(w, x)])
    y = dot + b[0]
    return y

def convert_model_to_zkp(weights_path, bias_path, metadata_out_path):
    weights, bias = load_model(weights_path, bias_path)
    commit_metadata(weights, bias, metadata_out_path)
    return weights, bias

# Example usage:
# from zkp.svm_zkp import convert_model_to_zkp, svm_predict_zkp
# weights, bias = convert_model_to_zkp('path_to_weights.npy', 'path_to_bias.npy', 'metadata_out.json')
# w = [PrivVal(int(x)) for x in weights.flatten()]
# b = [PrivVal(int(x)) for x in bias.flatten()]
# x = [PrivVal(int(f)) for f in sample_input]
# y = svm_predict_zkp(w, b, x)
