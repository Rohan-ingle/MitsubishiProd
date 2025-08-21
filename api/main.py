from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import json
import sys

# Ensure modules import works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.svm_zkp import convert_model_to_zkp
from api.federated_utils import train_bank_model, aggregate_master_model
from api.zkp_utils import classify_with_zkp, verify_zkp

app = FastAPI()

DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'archive', 'creditcard.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'master')
ZKP_DIR = os.path.join(PROJECT_ROOT, 'models', 'zkp')
DEPLOY_DIR = os.path.join(PROJECT_ROOT, 'deploy')

class TransactionRequest(BaseModel):
    sample_id: int

@app.get("/transactions/count")
def get_transaction_count():
    df = pd.read_csv(DATA_PATH)
    return {"count": len(df)}

@app.get("/transactions/{sample_id}")
def get_transaction(sample_id: int):
    df = pd.read_csv(DATA_PATH)
    if sample_id < 0 or sample_id >= len(df):
        raise HTTPException(status_code=404, detail="Sample not found")
    row = df.iloc[sample_id]
    return row.to_dict()


class ClassifyRequest(BaseModel):
    sample_id: int

@app.post("/bank1/classify")
def bank1_classify(req: ClassifyRequest):
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    y = df['Class']
    sample_id = req.sample_id
    if sample_id < 0 or sample_id >= len(df):
        raise HTTPException(status_code=404, detail="Sample not found")
    sample_features = X.iloc[sample_id].values
    true_label = int(y.iloc[sample_id])
    
    # Use our new ZKP implementation
    result = classify_with_zkp(sample_features)
    
    # Create message
    message = {
        'sample_id': int(sample_id),
        'prediction': result['prediction'],
        'true_label': true_label,
        'proof_path': result.get('proof_path', ''),
        'zkp_status': result['zkp_status']
    }
    
    # Save message for Bank2
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    message_path = os.path.join(DEPLOY_DIR, 'bank_message.json')
    with open(message_path, 'w') as f:
        json.dump(message, f, indent=2)
    
    return message


@app.get("/bank2/verify")
def bank2_verify():
    message_path = os.path.join(DEPLOY_DIR, 'bank_message.json')
    if not os.path.exists(message_path):
        raise HTTPException(status_code=404, detail="No message found")
    
    with open(message_path, 'r') as f:
        message = json.load(f)
    
    prediction = message['prediction']
    sample_id = message['sample_id']
    proof_path = message.get('proof_path')
    verified = False
    
    # Only try to verify if we have a proof path
    if proof_path and os.path.exists(proof_path):
        try:
            # Use our new ZKP verification
            verified = verify_zkp(proof_path)
            
            if verified:
                return {
                    'sample_id': sample_id,
                    'prediction': prediction,
                    'verified': True,
                    'true_label': message.get('true_label', None),
                    'status': "Verification successful: The classification result is trustworthy."
                }
            else:
                return {
                    'sample_id': sample_id,
                    'prediction': prediction,
                    'verified': False,
                    'true_label': message.get('true_label', None),
                    'status': "Verification failed: Cannot trust the classification result."
                }
        except Exception as e:
            return {
                'sample_id': sample_id,
                'prediction': prediction,
                'verified': False,
                'true_label': message.get('true_label', None),
                'status': f"Verification error: {str(e)}"
            }
    else:
        return {
            'sample_id': sample_id,
            'prediction': prediction,
            'verified': False,
            'true_label': message.get('true_label', None),
            'status': "Proof file not found"
        }
    
    return {
        'sample_id': sample_id,
        'prediction': prediction,
        'verified': verified,
        'true_label': message.get('true_label', None),
        'status': "Transaction processed normally."
    }


class FederatedTrainRequest(BaseModel):
    bank: str  # 'bank1' or 'bank2'

@app.post("/federated/train")
def federated_train(req: FederatedTrainRequest):
    bank = req.bank
    if bank not in ["bank1", "bank2"]:
        raise HTTPException(status_code=400, detail="Invalid bank name")
    model_dir = os.path.join(PROJECT_ROOT, 'models', bank)
    train_bank_model(bank, DATA_PATH, model_dir)
    return {"status": "trained", "bank": bank}


@app.post("/federated/aggregate")
def federated_aggregate():
    try:
        # Aggregate Bank1 and Bank2 models
        bank1_dir = os.path.join(PROJECT_ROOT, 'models', 'bank1')
        bank2_dir = os.path.join(PROJECT_ROOT, 'models', 'bank2')
        master_dir = os.path.join(PROJECT_ROOT, 'models', 'master')
        
        # Ensure the required files exist
        required_files = [
            os.path.join(bank1_dir, 'svm_weights_bank1.npy'),
            os.path.join(bank1_dir, 'svm_bias_bank1.npy'),
            os.path.join(bank2_dir, 'svm_weights_bank2.npy'),
            os.path.join(bank2_dir, 'svm_bias_bank2.npy')
        ]
        
        for f in required_files:
            if not os.path.exists(f):
                return {"status": "error", "message": f"Required file not found: {f}"}
        
        # Aggregate the models
        print(f"Aggregating models from {bank1_dir} and {bank2_dir}...")
        success = aggregate_master_model(bank1_dir, bank2_dir, master_dir)
        
        if not success:
            return {"status": "error", "message": "Model aggregation failed"}
        
        # Generate ZKP metadata for the master model
        weights_path = os.path.join(master_dir, 'svm_weights_master.npy')
        bias_path = os.path.join(master_dir, 'svm_bias_master.npy')
        zkp_dir = os.path.join(PROJECT_ROOT, 'models', 'zkp')
        os.makedirs(zkp_dir, exist_ok=True)
        zkp_metadata_path = os.path.join(zkp_dir, 'svm_master_zkp_metadata.json')
        
        print(f"Generating ZKP metadata for master model...")
        weights, bias = convert_model_to_zkp(weights_path, bias_path, zkp_metadata_path)
        
        # Verify that the aggregate model was created
        if not os.path.exists(weights_path) or not os.path.exists(bias_path):
            return {"status": "error", "message": "Failed to create aggregated model files"}
        
        # Create a test proof to verify everything works
        try:
            # Import our ZKP processor
            from zkpy_svm_flow import SVMZkpProcessor
            
            # Generate a random sample for testing
            sample_features = np.random.rand(weights.shape[1])
            
            # Initialize ZKP processor with master model
            zkp = SVMZkpProcessor(model_dir=master_dir, circuit_path=os.path.join(PROJECT_ROOT, 'circuit.r1cs'))
            
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
                    zkp_status = "ZKP test successful"
                else:
                    print(f"Warning: ZKP test proof was generated but verification failed")
                    zkp_status = "ZKP test generated but verification failed"
            else:
                print(f"Warning: ZKP test failed to generate a proof")
                zkp_status = "ZKP test failed to generate proof"
        except Exception as e:
            print(f"Warning: ZKP test encountered an error: {str(e)}")
            zkp_status = f"ZKP test error: {str(e)}"
        
        return {
            "status": "success", 
            "message": "Models aggregated and ZKP metadata generated",
            "zkp_status": zkp_status,
            "model_files": {
                "weights": weights_path,
                "bias": bias_path,
                "metadata": zkp_metadata_path
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Aggregation failed: {str(e)}"}
