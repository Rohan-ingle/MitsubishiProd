# Fraud Detection System with Zero Knowledge Proofs

## Real-Time Financial Fraud Detection System

Financial fraud is a growing concern, especially with the rise of digital payments and online banking. This system was developed to detect and flag suspicious transactions in real-time. The system:

- Learns from historical transaction data
- Identifies anomalies and patterns
- Alerts users and institutions instantly

This project implements a secure fraud detection system using machine learning with zero-knowledge proofs (ZKP). The system allows for privacy-preserving fraud classification by using Support Vector Machine (SVM) models wrapped in ZKP circuits.

## Prerequisites

- Python 3.8+
- Circom (for circuit compilation)
- Node.js (for Circom dependencies)
- See `requirements.txt` for all Python dependencies

## Project Structure

```
.
├── api/               # FastAPI server code (REST API for training, prediction, verification)
├── dataset/           # Credit card fraud dataset
│   └── archive/
│       └── creditcard.csv
├── models/            # Trained ML models and compiled circuits
│   ├── compiled/      # Compiled Circom circuits 
│   └── zkl/           # Generated ZK proofs and public inputs
├── train/             # Training scripts for the fraud detection model (bank1, bank2, master)
├── zkp/               # Zero-knowledge proof implementations and SVM ZKP scripts
├── streamlit_app.py   # Streamlit web UI for demo and interaction
├── requirements.txt   # Python dependencies
└── compile_circuit.py # Script to compile the Circom circuit
```

## Setup

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Install Circom following instructions at https://docs.circom.io/getting-started/installation/

3. Compile the ZKP circuit:

```bash
python compile_circuit.py
```


This will create the necessary `.r1cs`, `.wasm`, and `.sym` files in the `models/compiled` directory.


## Running the System

### 1. Start the API server (FastAPI):

```bash
cd api
uvicorn main:app --reload
```

### 2. Start the Streamlit web app:

```bash
streamlit run streamlit_app.py
```

### 3. Use the endpoints to:
   - Train models
   - Make predictions with ZKP
   - Verify ZKP proofs

## API Endpoints (FastAPI)

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

If you have issues with ZKP libraries (`zkpy`, `pysnark`, `zkml`), ensure they are installed and available in your environment. Some may require additional system dependencies or custom installation steps.

## ZKP Implementation

The system uses SVM (Support Vector Machine) for classification, wrapped in a Circom circuit for zero-knowledge proofs. The circuit:

1. Takes the model weights, bias, and input features
2. Computes the SVM prediction (dot product + bias)
3. Generates a proof that the computation was done correctly without revealing the input data

This allows verification that a sample was correctly classified without revealing the actual sample data.

---

**Note:** If you see your workspace associated with an unexpected GitHub repository, check for nested `.git` folders (e.g., in `qaptools_build/`). Your main project is not a git repository unless you initialize it at the root.
