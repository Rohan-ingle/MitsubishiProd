import os
import subprocess
import sys

def compile_circuit():
    # Define paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    circuit_src = os.path.join(project_root, 'zkp', 'svm.circ')
    compiled_dir = os.path.join(project_root, 'models', 'compiled')
    
    # Create compiled directory if it doesn't exist
    os.makedirs(compiled_dir, exist_ok=True)
    
    # Check if circom is installed
    try:
        subprocess.run(['circom', '--version'], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Circom is not installed or not in PATH.")
        print("Please install Circom from https://docs.circom.io/getting-started/installation/")
        sys.exit(1)
    
    # Compile the circuit
    print(f"Compiling circuit: {circuit_src}")
    compile_cmd = [
        'circom',
        circuit_src,
        '--r1cs',
        '--wasm',
        '--sym',
        '-o', compiled_dir
    ]
    
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Circuit compiled successfully!")
        print(f"Output files are in: {compiled_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error compiling circuit: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    compile_circuit()
