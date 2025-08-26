# Zero-Knowledge Proof System for Cross-Bank Fraud Detection

## Overview

This document provides detailed documentation for the Zero-Knowledge Proof (ZKP) system implemented in the MitsubishiProd project. The system enables secure fraud detection collaboration between banks without sharing sensitive data.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Implementation Details](#implementation-details)
4. [Workflow](#workflow)
5. [API Reference](#api-reference)
6. [Security Considerations](#security-considerations)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

## Introduction

### Purpose

The ZKP system allows two banks to collaborate on fraud detection without sharing:
- Proprietary fraud detection models
- Sensitive customer transaction data
- Private financial information

### Key Features

- **Privacy Preservation**: Banks can verify fraud classifications without revealing models or data
- **Cryptographic Security**: Uses zero-knowledge proofs to ensure verification without data exposure
- **Flexibility**: Supports both binary and JSON-based proofs
- **Fallback Mechanisms**: Gracefully handles environments where full ZKP functionality is unavailable

## Architecture

### Components

1. **ZKP Processor (`SVMZkpProcessor`)**: Core class handling circuit compilation, proof generation, and verification
2. **Circuit Representation**: R1CS files representing the SVM computation
3. **Proof Generation**: Creates cryptographic proofs of SVM classification correctness
4. **Verification System**: Verifies proofs without accessing private data

### Libraries

- **zkpy**: Primary ZKP library providing circuit, proving, and verification functionality
- **pysnark**: Secondary library used for model conversion to ZKP-compatible formats

### Directory Structure

```
- api/
  - zkp_utils.py        # API endpoints for ZKP functions
- models/
  - bank1/              # Bank1's SVM model
  - zkp/                # ZKP proof storage
  - zkl/                # JSON proof storage
- inferencing/
  - bank1_classify_and_send_zkpy.py     # Bank1's classification with ZKP
  - bank2_receive_and_verify_zkpy.py    # Bank2's verification
- zkpy_svm_flow.py      # Core ZKP implementation
- zkp/
  - compile_svm_circuit.py    # Circuit compilation
  - convert_master_to_zkp.py  # Model conversion for ZKP
```

## Implementation Details

### SVM Model and Circuit

The system uses a Support Vector Machine (SVM) model for fraud detection, represented as:

```
prediction = (dot_product(weights, features) + bias) > 0
```

This computation is encoded in a circuit representation (R1CS format) that allows zero-knowledge proofs to be generated and verified.

### Witness Structure

The witness (private information) consists of:

1. **Model Weights**: The SVM model weights
2. **Bias Term**: The SVM bias value
3. **Transaction Features**: The features of the transaction being classified

```python
witness = weights.flatten().tolist() + [float(bias)] + features.tolist()
```

### Proof Generation

Proofs are generated through the following process:

1. **Circuit Compilation**: Load/compile the SVM circuit
2. **Witness Creation**: Combine model parameters and transaction features
3. **Hash Generation**: Create cryptographic hashes of witness and circuit
4. **Proof Creation**: Generate a ZKP using the zkpy library
5. **JSON Storage**: Store proof data including cryptographic hashes

```python
# Generate cryptographic proof
circuit.prove(witness, proof_path)
```

### Verification Process

Verification occurs in multiple layers:

1. **Native ZKP Verification**: Using zkpy's verify method
   ```python
   verification_result = circuit.verify(proof_path)
   ```

2. **Hash Verification**: Checking cryptographic hashes of circuit and witness
   ```python
   calculated_circuit_hash = hashlib.sha256(circuit_content).hexdigest()
   if calculated_circuit_hash != proof_data['circuit_hash']:
       return False
   ```

3. **Structure Verification**: Ensuring all required cryptographic elements are present

## Workflow

### Bank-to-Bank Fraud Detection Flow

1. **Model Preparation**:
   - Bank1 trains an SVM model on their fraud detection dataset
   - Circuit is compiled to represent the SVM computation

2. **Transaction Classification**:
   - Bank1 extracts features from a transaction
   - Bank1 classifies the transaction using their SVM model
   - Bank1 generates a ZKP proving the classification is correct

3. **Proof Sharing**:
   - Bank1 sends the classification result and proof to Bank2
   - No model parameters or transaction details are shared

4. **Verification**:
   - Bank2 loads the same circuit (but not the witness)
   - Bank2 verifies the proof cryptographically
   - If verified, Bank2 can trust the classification without seeing Bank1's data

5. **Action**:
   - Bank2 takes appropriate action based on the verified classification
   - For fraud: flag for investigation
   - For normal: process normally

## API Reference

### SVMZkpProcessor Class

The core class implementing ZKP functionality.

#### Constructor

```python
def __init__(self, model_dir=None, circuit_path=None, compiled_dir=None):
    """
    Initialize the SVM ZKP processor.
    
    Args:
        model_dir: Directory containing the model weights and bias
        circuit_path: Path to the SVM circuit file
        compiled_dir: Directory to store compiled circuit artifacts
    """
```

#### Methods

##### compile_circuit

```python
def compile_circuit(self):
    """Initialize the Circuit object with the pre-compiled circuit."""
```

##### create_witness

```python
def create_witness(self, weights, bias, x_features):
    """
    Create a witness for the SVM circuit.
    
    Args:
        weights: Model weights (numpy array)
        bias: Model bias (float)
        x_features: Input features to classify (numpy array)
            
    Returns:
        List of witness values
    """
```

##### generate_proof

```python
def generate_proof(self, weights, bias, x_features, proof_path=None):
    """
    Generate a ZK proof for the SVM classification.
    
    Args:
        weights: Model weights (numpy array)
        bias: Model bias (float)
        x_features: Input features to classify (numpy array)
        proof_path: Path to save the proof
            
    Returns:
        Path to the saved proof
    """
```

##### verify_proof

```python
def verify_proof(self, proof_path):
    """
    Verify a ZK proof.
    
    Args:
        proof_path: Path to the proof file (either JSON or zkpy format)
            
    Returns:
        Boolean indicating if the proof is valid
    """
```

##### verify_zkpy_proof

```python
def verify_zkpy_proof(self, proof_path):
    """
    Verify a zkpy proof.
    
    Args:
        proof_path: Path to the zkpy proof file
            
    Returns:
        Boolean indicating if the proof is valid
    """
```

##### load_model

```python
def load_model(self, weights_path=None, bias_path=None):
    """
    Load SVM model weights and bias.
    
    Args:
        weights_path: Path to the weights file (.npy)
        bias_path: Path to the bias file (.npy)
            
    Returns:
        Tuple of (weights, bias)
    """
```

### Helper Functions

#### create_bank_message

```python
def create_bank_message(sample_id, prediction, proof_path, y_true=None):
    """
    Create a message to send from Bank1 to Bank2.
    
    Args:
        sample_id: ID of the sample being classified
        prediction: Classification result (0=normal, 1=fraud)
        proof_path: Path to the ZKP proof file
        y_true: True label (if available)
        
    Returns:
        Dictionary containing the message
    """
```

### API Endpoints (zkp_utils.py)

#### classify_with_zkp

```python
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
```

#### verify_zkp

```python
def verify_zkp(proof_path):
    """
    Verify a ZKP proof.
    
    Args:
        proof_path: Path to the proof file
        
    Returns:
        Boolean indicating if the proof is valid
    """
```

## Security Considerations

### Cryptographic Security

The system provides these security properties:

1. **Zero-Knowledge**: No information about the private inputs (model parameters, transaction features) is revealed during verification.

2. **Soundness**: It is computationally infeasible to generate a valid proof for an incorrect computation.

3. **Completeness**: Correct computations can always be proven.

### Hash Security

The system uses SHA-256 for cryptographic hashing, providing:

- 128-bit security level
- Collision resistance
- Pre-image resistance

### Fallback Security

When full ZKP verification is unavailable, the fallback provides:

- Circuit integrity verification
- Format validation of cryptographic elements
- Basic hash validation

Note: The fallback provides less cryptographic assurance than full ZKP verification.

## Usage Examples

### Bank1: Classifying and Sending Proof

```python
# Initialize ZKP processor
zkp = SVMZkpProcessor(model_dir=BANK1_MODEL_DIR)

# Compile circuit
zkp.compile_circuit()

# Load model
weights, bias = zkp.load_model()

# Generate proof for a transaction
result = zkp.generate_proof(weights, bias, transaction_features)

# Create message for Bank2
create_bank_message(transaction_id, result['prediction'], result['proof_path'])
```

### Bank2: Receiving and Verifying

```python
# Load message from Bank1
with open(MESSAGE_PATH, 'r') as f:
    message = json.load(f)

sample_id = message['sample_id']
prediction = message['prediction']
proof_path = message['proof_path']

# Initialize ZKP processor
zkp = SVMZkpProcessor()

# Compile circuit
zkp.compile_circuit()

# Verify proof
verified = zkp.verify_proof(proof_path)

# Take action based on verification
if verified:
    if prediction == 1:
        # Handle fraud case
    else:
        # Handle normal case
else:
    # Handle verification failure
```

## Troubleshooting

### Common Issues

1. **Missing zkpy Library**:
   - Error: `ImportError: No module named 'zkpy'`
   - Solution: Install the zkpy library using `pip install zkpy`

2. **Circuit Compilation Failure**:
   - Error: `FileNotFoundError: Circuit file not found`
   - Solution: Ensure the circuit file exists at the specified path

3. **Verification Failure**:
   - Error: `Circuit hash mismatch`
   - Solution: Ensure both banks are using the same circuit file

4. **JSON Proof Missing Fields**:
   - Error: `JSON proof missing required fields`
   - Solution: Ensure the proof generation includes all required cryptographic fields

### Debugging

Enable detailed logging by adding print statements:

```python
# Debug proof generation
print(f"Generating proof with witness: {witness[:5]}...")
print(f"Using circuit: {self.circuit_path}")

# Debug verification
print(f"Verifying proof: {proof_path}")
print(f"Circuit hash: {calculated_circuit_hash}")
```