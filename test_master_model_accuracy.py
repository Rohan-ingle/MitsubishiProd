"""
Master Model Accuracy Test Script

This script evaluates the accuracy of the master model on test data from both banks.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Ensure proper import paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def evaluate_master_model():
    """
    Evaluate the accuracy of the master model on test data from both banks.
    """
    print("Evaluating Master Model Accuracy...")
    
    # Define paths
    master_dir = os.path.join(PROJECT_ROOT, 'models', 'master')
    weights_path = os.path.join(master_dir, 'svm_weights_master.npy')
    bias_path = os.path.join(master_dir, 'svm_bias_master.npy')
    data_path = os.path.join(PROJECT_ROOT, 'dataset', 'archive', 'creditcard.csv')
    
    # Check if the master model exists
    if not os.path.exists(weights_path) or not os.path.exists(bias_path):
        print("Error: Master model files not found. Please run the master aggregator first.")
        return False
    
    # Load master model
    try:
        weights = np.load(weights_path)
        bias = np.load(bias_path)
        print(f"Master model loaded: weights shape {weights.shape}, bias {float(bias[0])}")
    except Exception as e:
        print(f"Error loading master model: {e}")
        return False
    
    # Load dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {len(df)} transactions with {df['Class'].sum()} frauds ({df['Class'].mean()*100:.2f}%)")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Split data into train and test sets (to match the training process)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Create test sets for each bank (to match bank1_train.py and bank2_train.py)
    mid = len(df) // 2
    
    # Bank 1 data (first half)
    X_bank1 = X.iloc[:mid]
    y_bank1 = y.iloc[:mid]
    X_train_bank1, X_test_bank1, y_train_bank1, y_test_bank1 = train_test_split(
        X_bank1, y_bank1, test_size=0.2, random_state=42, stratify=y_bank1
    )
    
    # Bank 2 data (second half)
    X_bank2 = X.iloc[mid:]
    y_bank2 = y.iloc[mid:]
    X_train_bank2, X_test_bank2, y_train_bank2, y_test_bank2 = train_test_split(
        X_bank2, y_bank2, test_size=0.2, random_state=42, stratify=y_bank2
    )
    
    # Combine test sets from both banks
    X_test_combined = pd.concat([X_test_bank1, X_test_bank2])
    y_test_combined = pd.concat([y_test_bank1, y_test_bank2])
    
    print(f"Testing on combined test data: {len(X_test_combined)} samples")
    
    # Define prediction function using the master model
    def predict(X):
        """Make predictions using the master model."""
        # Flatten weights if needed
        if len(weights.shape) > 1:
            w = weights.flatten()
        else:
            w = weights
            
        # Calculate predictions
        predictions = []
        for i in range(len(X)):
            # Calculate decision value: dot product + bias
            decision_value = np.dot(w, X.iloc[i].values) + bias[0]
            # Class prediction: 1 if decision_value > 0, else 0
            prediction = 1 if decision_value > 0 else 0
            predictions.append(prediction)
        return np.array(predictions)
    
    # Make predictions on the test sets
    try:
        y_pred_bank1 = predict(X_test_bank1)
        y_pred_bank2 = predict(X_test_bank2)
        y_pred_combined = predict(X_test_combined)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return False
    
    # Calculate metrics for Bank 1 test data
    accuracy_bank1 = accuracy_score(y_test_bank1, y_pred_bank1)
    precision_bank1 = precision_score(y_test_bank1, y_pred_bank1, zero_division=0)
    recall_bank1 = recall_score(y_test_bank1, y_pred_bank1, zero_division=0)
    f1_bank1 = f1_score(y_test_bank1, y_pred_bank1, zero_division=0)
    cm_bank1 = confusion_matrix(y_test_bank1, y_pred_bank1)
    
    # Calculate metrics for Bank 2 test data
    accuracy_bank2 = accuracy_score(y_test_bank2, y_pred_bank2)
    precision_bank2 = precision_score(y_test_bank2, y_pred_bank2, zero_division=0)
    recall_bank2 = recall_score(y_test_bank2, y_pred_bank2, zero_division=0)
    f1_bank2 = f1_score(y_test_bank2, y_pred_bank2, zero_division=0)
    cm_bank2 = confusion_matrix(y_test_bank2, y_pred_bank2)
    
    # Calculate metrics for combined test data
    accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
    precision_combined = precision_score(y_test_combined, y_pred_combined, zero_division=0)
    recall_combined = recall_score(y_test_combined, y_pred_combined, zero_division=0)
    f1_combined = f1_score(y_test_combined, y_pred_combined, zero_division=0)
    cm_combined = confusion_matrix(y_test_combined, y_pred_combined)
    
    # Print results
    print("\n=== Master Model Evaluation Results ===")
    
    print("\nBank 1 Test Data:")
    print(f"Accuracy: {accuracy_bank1:.4f}")
    print(f"Precision: {precision_bank1:.4f}")
    print(f"Recall: {recall_bank1:.4f}")
    print(f"F1 Score: {f1_bank1:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {cm_bank1[0, 0]}, FP: {cm_bank1[0, 1]}")
    print(f"FN: {cm_bank1[1, 0]}, TP: {cm_bank1[1, 1]}")
    
    print("\nBank 2 Test Data:")
    print(f"Accuracy: {accuracy_bank2:.4f}")
    print(f"Precision: {precision_bank2:.4f}")
    print(f"Recall: {recall_bank2:.4f}")
    print(f"F1 Score: {f1_bank2:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {cm_bank2[0, 0]}, FP: {cm_bank2[0, 1]}")
    print(f"FN: {cm_bank2[1, 0]}, TP: {cm_bank2[1, 1]}")
    
    print("\nCombined Test Data:")
    print(f"Accuracy: {accuracy_combined:.4f}")
    print(f"Precision: {precision_combined:.4f}")
    print(f"Recall: {recall_combined:.4f}")
    print(f"F1 Score: {f1_combined:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {cm_combined[0, 0]}, FP: {cm_combined[0, 1]}")
    print(f"FN: {cm_combined[1, 0]}, TP: {cm_combined[1, 1]}")
    
    # Also evaluate against the full dataset
    print("\nEvaluation on Full Dataset:")
    try:
        y_pred_full = predict(X)
        accuracy_full = accuracy_score(y, y_pred_full)
        precision_full = precision_score(y, y_pred_full, zero_division=0)
        recall_full = recall_score(y, y_pred_full, zero_division=0)
        f1_full = f1_score(y, y_pred_full, zero_division=0)
        cm_full = confusion_matrix(y, y_pred_full)
        
        print(f"Accuracy: {accuracy_full:.4f}")
        print(f"Precision: {precision_full:.4f}")
        print(f"Recall: {recall_full:.4f}")
        print(f"F1 Score: {f1_full:.4f}")
        print("Confusion Matrix:")
        print(f"TN: {cm_full[0, 0]}, FP: {cm_full[0, 1]}")
        print(f"FN: {cm_full[1, 0]}, TP: {cm_full[1, 1]}")
    except Exception as e:
        print(f"Error evaluating on full dataset: {e}")
    
    print("\nMaster Model Evaluation Complete!")
    return True

def compare_with_individual_models():
    """
    Compare the master model with individual bank models.
    """
    print("\n=== Comparing Master Model with Individual Bank Models ===")
    
    # Define paths
    master_dir = os.path.join(PROJECT_ROOT, 'models', 'master')
    bank1_dir = os.path.join(PROJECT_ROOT, 'models', 'bank1')
    bank2_dir = os.path.join(PROJECT_ROOT, 'models', 'bank2')
    
    # Check if all models exist
    master_weights_path = os.path.join(master_dir, 'svm_weights_master.npy')
    master_bias_path = os.path.join(master_dir, 'svm_bias_master.npy')
    bank1_weights_path = os.path.join(bank1_dir, 'svm_weights_bank1.npy')
    bank1_bias_path = os.path.join(bank1_dir, 'svm_bias_bank1.npy')
    bank2_weights_path = os.path.join(bank2_dir, 'svm_weights_bank2.npy')
    bank2_bias_path = os.path.join(bank2_dir, 'svm_bias_bank2.npy')
    
    required_files = [
        master_weights_path, master_bias_path,
        bank1_weights_path, bank1_bias_path,
        bank2_weights_path, bank2_bias_path
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            print("Please make sure all models are trained and aggregated.")
            return False
    
    # Load models
    try:
        master_weights = np.load(master_weights_path)
        master_bias = np.load(master_bias_path)
        bank1_weights = np.load(bank1_weights_path)
        bank1_bias = np.load(bank1_bias_path)
        bank2_weights = np.load(bank2_weights_path)
        bank2_bias = np.load(bank2_bias_path)
        
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
    
    # Compare model parameters
    print("\nModel Parameters Comparison:")
    print(f"Master model bias: {float(master_bias[0]):.6f}")
    print(f"Bank1 model bias: {float(bank1_bias[0]):.6f}")
    print(f"Bank2 model bias: {float(bank2_bias[0]):.6f}")
    
    # Calculate weight differences
    master_bank1_diff = np.mean(np.abs(master_weights - bank1_weights))
    master_bank2_diff = np.mean(np.abs(master_weights - bank2_weights))
    bank1_bank2_diff = np.mean(np.abs(bank1_weights - bank2_weights))
    
    print(f"\nAverage weight difference:")
    print(f"Master vs Bank1: {master_bank1_diff:.6f}")
    print(f"Master vs Bank2: {master_bank2_diff:.6f}")
    print(f"Bank1 vs Bank2: {bank1_bank2_diff:.6f}")
    
    # Check if master is closer to the average of bank1 and bank2
    avg_weights = (bank1_weights + bank2_weights) / 2
    master_avg_diff = np.mean(np.abs(master_weights - avg_weights))
    
    print(f"\nMaster vs Average of Bank1 & Bank2: {master_avg_diff:.6f}")
    
    if master_avg_diff < 1e-5:
        print("Master model is correctly computing the average of Bank1 and Bank2 models.")
    else:
        print("Note: Master model differs from the simple average of Bank1 and Bank2 models.")
    
    return True

if __name__ == "__main__":
    # Evaluate master model accuracy
    evaluate_master_model()
    
    # Compare with individual models
    compare_with_individual_models()
