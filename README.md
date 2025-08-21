# Fraud Detection System with Zero Knowledge Proofs

This project implements a secure fraud detection system using machine learning with zero-knowledge proofs (ZKP). The system allows for privacy-preserving fraud classification by using Support Vector Machine (SVM) models wrapped in ZKP circuits.

## Prerequisites

- Python 3.8+
- Circom (for circuit compilation)
- Node.js (for Circom dependencies)

## Project Structure

```
.
├── api/               # FastAPI server code
├── dataset/           # Credit card fraud dataset
│   └── archive/
│       └── creditcard.csv
├── models/            # Trained ML models and compiled circuits
│   ├── compiled/      # Compiled Circom circuits 
│   └── zkl/           # Generated ZK proofs
├── train/             # Training scripts for the fraud detection model
├── zkp/               # Zero-knowledge proof implementations
└── compile_circuit.py # Script to compile the Circom circuit
```

## Setup

1. Install required Python packages:

```bash
pip install fastapi uvicorn numpy pandas zkpy circompy
```

2. Install Circom following instructions at https://docs.circom.io/getting-started/installation/

3. Compile the ZKP circuit:

```bash
python compile_circuit.py
```

This will create the necessary `.r1cs`, `.wasm`, and `.sym` files in the `models/compiled` directory.

## Running the System

1. Start the API server:

```bash
cd api
uvicorn main:app --reload
```

2. Use the endpoints to:
   - Train models
   - Make predictions with ZKP

## API Endpoints

- `POST /train`: Train the fraud detection model
- `POST /predict`: Make a prediction with zero-knowledge proof
- `POST /verify`: Verify a zero-knowledge proof

## Circuit Compilation

The system uses Circom to compile circuits. If you get an error about missing compiled circuits, run:

```bash
python compile_circuit.py
```

## Troubleshooting

If you encounter the error "Circuit not compiled", make sure you've run the circuit compilation step:

```bash
python compile_circuit.py
```

If Circom is not installed or not in your PATH, you'll need to install it following the instructions at https://docs.circom.io/getting-started/installation/

## ZKP Implementation

The system uses SVM (Support Vector Machine) for classification, wrapped in a Circom circuit for zero-knowledge proofs. The circuit:

1. Takes the model weights, bias, and input features
2. Computes the SVM prediction (dot product + bias)
3. Generates a proof that the computation was done correctly without revealing the input data

This allows verification that a sample was correctly classified without revealing the actual sample data.
