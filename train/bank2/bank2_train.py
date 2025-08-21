
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample
import joblib
import numpy as np
import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'archive', 'creditcard.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'bank2')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset and split for bank2
df = pd.read_csv(DATA_PATH)
mid = len(df) // 2
# df_bank2 = df.iloc[mid:]
df_bank2 = df
X = df_bank2.drop('Class', axis=1)
y = df_bank2['Class']

# Train-test split (bank2 local)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply class balancing to address severe imbalance (0.17% fraud)
print(f"Before balancing - Training samples: {len(X_train)}, Fraud samples: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.2f}%)")

# Separate majority and minority classes
X_train_df = X_train.copy()
X_train_df['Class'] = y_train.values
fraud_samples = X_train_df[X_train_df['Class'] == 1]
non_fraud_samples = X_train_df[X_train_df['Class'] == 0]

# Downsample majority class or upsample minority class
# Option 1: Downsample majority class
# We'll downsample to a more balanced ratio (e.g., 50:50 or 80:20)
n_non_fraud = len(fraud_samples) * 5  # 80:20 ratio
non_fraud_downsampled = resample(
    non_fraud_samples, 
    replace=False,
    n_samples=n_non_fraud,
    random_state=42
)

# Combine downsampled majority with minority class
balanced_df = pd.concat([non_fraud_downsampled, fraud_samples])

# Prepare balanced training data
X_train_balanced = balanced_df.drop('Class', axis=1)
y_train_balanced = balanced_df['Class']

print(f"After balancing - Training samples: {len(X_train_balanced)}, Fraud samples: {sum(y_train_balanced)} ({sum(y_train_balanced)/len(y_train_balanced)*100:.2f}%)")

# Train linear SVM for 10 epochs with balanced data
model = SGDClassifier(loss='hinge', max_iter=10, tol=1e-3, random_state=42, class_weight='balanced')
model.fit(X_train_balanced, y_train_balanced)

# Save local model
joblib.dump(model, os.path.join(MODEL_DIR, 'svm_model_bank2.joblib'))
np.save(os.path.join(MODEL_DIR, 'svm_weights_bank2.npy'), model.coef_)
np.save(os.path.join(MODEL_DIR, 'svm_bias_bank2.npy'), model.intercept_)
