import sys
import os
import inspect
import zkpy

# Ensure modules import works
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Create a dummy circuit
dummy_path = os.path.join(PROJECT_ROOT, 'dummy.r1cs')
with open(dummy_path, 'w') as f:
    f.write("dummy")

# Try to get the method signatures
circuit = zkpy.Circuit(dummy_path)

# Clean up
os.remove(dummy_path)

# Inspect the prove method
prove_sig = inspect.signature(circuit.prove)
print(f"circuit.prove signature: {prove_sig}")
print(f"Parameters: {prove_sig.parameters}")

# Inspect the verify method
verify_sig = inspect.signature(circuit.verify)
print(f"\ncircuit.verify signature: {verify_sig}")
print(f"Parameters: {verify_sig.parameters}")
