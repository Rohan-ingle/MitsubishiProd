import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import joblib

def train_bank_model(bank: str, data_path: str, model_dir: str):
    df = pd.read_csv(data_path)
    mid = len(df) // 2
    if bank == 'bank1':
        df_bank = df.iloc[:mid]
    elif bank == 'bank2':
        df_bank = df.iloc[mid:]
    else:
        raise ValueError('bank must be "bank1" or "bank2"')
    X = df_bank.drop('Class', axis=1)
    y = df_bank['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = SGDClassifier(loss='hinge', max_iter=10, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f'svm_model_{bank}.joblib'))
    np.save(os.path.join(model_dir, f'svm_weights_{bank}.npy'), model.coef_)
    np.save(os.path.join(model_dir, f'svm_bias_{bank}.npy'), model.intercept_)
    return True

def find_optimal_bias(weights, data_path, target_metric='f1'):
    """
    Find the optimal bias term for a given weight vector and dataset.
    
    Args:
        weights: The weight vector (coefficients) of the SVM model
        data_path: Path to the dataset CSV file
        target_metric: Metric to optimize for (default: f1 score)
        
    Returns:
        An optimized bias term as a numpy array with shape (1,)
    """
    print(f"Finding optimal bias using {target_metric} score as target...")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Create train/test split (to match the training process)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Function to calculate predictions with a given bias
    def predict_with_bias(X_data, bias_value):
        predictions = []
        for i in range(len(X_data)):
            decision_value = np.dot(weights.flatten(), X_data.iloc[i].values) + bias_value
            prediction = 1 if decision_value > 0 else 0
            predictions.append(prediction)
        return np.array(predictions)
    
    # Search for optimal bias
    best_score = -1
    best_bias = 0
    
    # Try a wider range of bias values with finer granularity 
    # for severely imbalanced dataset
    bias_range = np.linspace(-30, 30, 500)
    
    print(f"Testing {len(bias_range)} bias values from {min(bias_range)} to {max(bias_range)}")
    
    for bias in bias_range:
        y_pred = predict_with_bias(X_test, bias)
        
        # Calculate the target metric
        if target_metric == 'f1':
            score = f1_score(y_test, y_pred, zero_division=0)
        elif target_metric == 'balanced_accuracy':
            # Balanced accuracy is the average of recall obtained on each class
            # Better for imbalanced datasets
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(y_test, y_pred)
        elif target_metric == 'gmean':
            # Geometric mean of recall and specificity
            # Good for imbalanced data
            from sklearn.metrics import recall_score, confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            recall = recall_score(y_test, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = np.sqrt(recall * specificity) if (recall * specificity) > 0 else 0
        else:
            # Default to F1 score if metric not recognized
            score = f1_score(y_test, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_bias = bias
            
        # Print progress every 100 steps
        if list(bias_range).index(bias) % 100 == 0:
            print(f"Progress: {list(bias_range).index(bias)/len(bias_range)*100:.1f}%, Current best bias: {best_bias} ({target_metric}: {best_score:.4f})")
    
    # Additional metrics for the best bias
    y_pred_best = predict_with_bias(X_test, best_bias)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best, zero_division=0)
    recall = recall_score(y_test, y_pred_best, zero_division=0)
    f1 = f1_score(y_test, y_pred_best, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    
    print(f"\nOptimal bias found: {best_bias} with {target_metric} score: {best_score:.4f}")
    print(f"Metrics with optimal bias:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Return as numpy array with shape (1,) to match the original bias format
    return np.array([best_bias])

def aggregate_master_model(bank1_dir: str, bank2_dir: str, master_dir: str, optimize_bias=True, data_path=None, target_metric='f1'):
    w1 = np.load(os.path.join(bank1_dir, 'svm_weights_bank1.npy'))
    b1 = np.load(os.path.join(bank1_dir, 'svm_bias_bank1.npy'))
    w2 = np.load(os.path.join(bank2_dir, 'svm_weights_bank2.npy'))
    b2 = np.load(os.path.join(bank2_dir, 'svm_bias_bank2.npy'))
    w_avg = (w1 + w2) / 2
    
    # Calculate initial bias as average of both models
    b_avg = (b1 + b2) / 2
    
    # If optimize_bias is True and a data path is provided, find an optimal bias
    if optimize_bias and data_path:
        try:
            b_avg = find_optimal_bias(w_avg, data_path, target_metric=target_metric)
            print(f"Optimized bias: {float(b_avg[0])}")
        except Exception as e:
            print(f"Error optimizing bias, using average instead: {e}")
    
    os.makedirs(master_dir, exist_ok=True)
    np.save(os.path.join(master_dir, 'svm_weights_master.npy'), w_avg)
    np.save(os.path.join(master_dir, 'svm_bias_master.npy'), b_avg)
    return True
