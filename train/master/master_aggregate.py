import numpy as np
import os
import sys
import subprocess


# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bank1_weights_path = os.path.join(BASE_DIR, 'models', 'bank1', 'svm_weights_bank1.npy')
bank1_bias_path = os.path.join(BASE_DIR, 'models', 'bank1', 'svm_bias_bank1.npy')
bank2_weights_path = os.path.join(BASE_DIR, 'models', 'bank2', 'svm_weights_bank2.npy')
bank2_bias_path = os.path.join(BASE_DIR, 'models', 'bank2', 'svm_bias_bank2.npy')
MASTER_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'master')
os.makedirs(MASTER_MODEL_DIR, exist_ok=True)

# Add BASE_DIR to sys.path
sys.path.insert(0, os.path.join(BASE_DIR))


# Load weights and biases
w1 = np.load(bank1_weights_path)
b1 = np.load(bank1_bias_path)
w2 = np.load(bank2_weights_path)
b2 = np.load(bank2_bias_path)

# Federated averaging
w_avg = (w1 + w2) / 2
b_avg = (b1 + b2) / 2

# Save aggregated model weights
np.save(os.path.join(MASTER_MODEL_DIR, 'svm_weights_master.npy'), w_avg)
np.save(os.path.join(MASTER_MODEL_DIR, 'svm_bias_master.npy'), b_avg)

print('Federated averaging complete. Aggregated weights and bias saved.')

# Compile SVM circuit using circom CLI
CIRCUIT_SRC = os.path.join(BASE_DIR, 'zkp', 'svm.circ')
COMPILED_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'compiled')
os.makedirs(COMPILED_MODEL_DIR, exist_ok=True)
CIRCOM_BIN = r"C:\Users\Rohan\AppData\Roaming\npm\circom.cmd"
subprocess.run([
    CIRCOM_BIN, CIRCUIT_SRC, "--r1cs", "--wasm", "--sym", "-o", COMPILED_MODEL_DIR
], check=True)
print(f'Compiled SVM circuit saved to {COMPILED_MODEL_DIR}')
