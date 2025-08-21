import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_DIR = os.path.join(BASE_DIR, 'deploy')
MESSAGE_PATH = os.path.join(DEPLOY_DIR, 'bank_message.json')

# Simulate receiving by reading the file

if not os.path.exists(MESSAGE_PATH):
    raise FileNotFoundError(f"Message file not found at {MESSAGE_PATH}")
with open(MESSAGE_PATH, 'r') as f:
    message = json.load(f)

sample_id = message['sample_id']
prediction = message['prediction']
y_value = message['y_value']

# In a real ZKP, Bank2 would verify the cryptographic proof. Here, we just check the structure.

verified = isinstance(y_value, int)

print(f"Bank2: Received transaction #{sample_id} with prediction: {prediction}")
if verified:
    print("Bank2: ZKP proof structure is valid. Accepting the result.")
    if prediction == 1:
        print("Bank2: Transaction flagged as FRAUD.")
    else:
        print("Bank2: Transaction is NOT fraud.")
else:
    print("Bank2: Invalid proof. Rejecting the result.")
