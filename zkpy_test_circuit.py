import sys
import os
import zkpy

# Ensure modules import works
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Create sample Circuit
circ_file = os.path.join(PROJECT_ROOT, 'models', 'compiled', 'svm.r1cs')
if os.path.exists(circ_file):
    try:
        # Try creating a circuit
        circuit = zkpy.Circuit(circ_file)
        print(f"Created circuit from {circ_file}")
        
        # Try using zkpy.prove if it exists
        if hasattr(zkpy, 'prove'):
            print("zkpy has 'prove' function")
        
        # Try using zkpy.verify if it exists
        if hasattr(zkpy, 'verify'):
            print("zkpy has 'verify' function")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Circuit file not found: {circ_file}")
    print("Looking for available circuit files...")
    compiled_dir = os.path.join(PROJECT_ROOT, 'models', 'compiled')
    if os.path.exists(compiled_dir):
        files = os.listdir(compiled_dir)
        print(f"Files in {compiled_dir}: {files}")
    else:
        print(f"Compiled directory not found: {compiled_dir}")
