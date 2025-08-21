
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import argparse
from pysnark.runtime import PrivVal, snark
import json
from modules.svm_zkp import commit_metadata, svm_predict_zkp

def load_model(weights_path, bias_path):
    weights = np.load(weights_path)
    bias = np.load(bias_path)
    return weights, bias


def main():
    parser = argparse.ArgumentParser(description='Convert SVM master model to ZKP using PySNARK')
    parser.add_argument('--weights', type=str, required=False, default=None, help='Path to SVM weights .npy file')
    parser.add_argument('--bias', type=str, required=False, default=None, help='Path to SVM bias .npy file')
    parser.add_argument('--out', type=str, required=False, default=None, help='Output path for ZKP metadata json')
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = args.weights or os.path.join(BASE_DIR, 'models', 'master', 'svm_weights_master.npy')
    bias_path = args.bias or os.path.join(BASE_DIR, 'models', 'master', 'svm_bias_master.npy')
    out_path = args.out or os.path.join(BASE_DIR, 'models', 'zkp', 'svm_master_zkp_metadata.json')
    zkl_dir = os.path.join(BASE_DIR, 'models', 'zkl')
    os.makedirs(zkl_dir, exist_ok=True)

    # Change working directory for PySNARK output
    os.chdir(zkl_dir)

    weights, bias = load_model(weights_path, bias_path)
    commit_metadata(weights, bias, out_path)

    # Generate a sample input
    num_features = weights.shape[1]
    sample_input = np.random.randint(0, 10, size=(num_features,))

    # Convert to PrivVal for ZKP
    w = [PrivVal(int(x)) for x in weights.flatten()]
    b = [PrivVal(int(x)) for x in bias.flatten()]
    x = [PrivVal(int(f)) for f in sample_input]

    print("Generating ZKP for SVM prediction (y >= 0)...")
    y = svm_predict_zkp(w, b, x)
    print(f"ZKP generated for y = {y}")
    print("You can now use PySNARK tools to export and verify the proof.")

if __name__ == "__main__":
    main()
