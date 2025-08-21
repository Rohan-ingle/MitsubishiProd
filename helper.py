import os
import subprocess
import sys
import argparse
import json

def setup_project():
    """Set up directories and check dependencies"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Create necessary directories
    directories = [
        os.path.join(project_root, 'models', 'compiled'),
        os.path.join(project_root, 'models', 'zkl'),
        os.path.join(project_root, 'api', 'deploy')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")
    
    # Check for Python dependencies
    try:
        import fastapi
        import numpy
        import pandas
        print("✓ Python dependencies verified")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Please run: pip install fastapi uvicorn numpy pandas")
        return False
    
    # Check for Circom
    try:
        result = subprocess.run(['circom', '--version'], 
                               check=True, capture_output=True, text=True)
        print(f"✓ Circom installed: {result.stdout.strip()}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Circom not found")
        print("Please install Circom: https://docs.circom.io/getting-started/installation/")
        return False
    
    return True

def compile_circuit():
    """Compile the SVM circuit using Circom"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    circuit_src = os.path.join(project_root, 'zkp', 'svm.circ')
    compiled_dir = os.path.join(project_root, 'models', 'compiled')
    
    # Create compiled directory if it doesn't exist
    os.makedirs(compiled_dir, exist_ok=True)
    
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
        print("✓ Circuit compiled successfully!")
        print(f"  Output files are in: {compiled_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error compiling circuit: {e}")
        if e.stdout:
            print(f"  Stdout: {e.stdout}")
        if e.stderr:
            print(f"  Stderr: {e.stderr}")
        return False

def start_api():
    """Start the FastAPI server"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    api_dir = os.path.join(project_root, 'api')
    
    try:
        os.chdir(api_dir)
        print(f"Starting API server in {api_dir}")
        subprocess.run(['uvicorn', 'main:app', '--reload'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API server: {e}")
        return False
    except FileNotFoundError:
        print("❌ uvicorn not found. Please install with: pip install uvicorn")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection ZKP System Helper")
    parser.add_argument('command', choices=['setup', 'compile', 'api'],
                        help='Command to run (setup, compile, api)')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        if setup_project():
            print("\n✓ Project setup complete")
        else:
            print("\n❌ Project setup incomplete. Please fix the issues above.")
            
    elif args.command == 'compile':
        if compile_circuit():
            print("\n✓ Circuit compilation complete")
        else:
            print("\n❌ Circuit compilation failed. Please fix the issues above.")
            
    elif args.command == 'api':
        # First check if the circuit is compiled
        project_root = os.path.dirname(os.path.abspath(__file__))
        r1cs_file = os.path.join(project_root, 'models', 'compiled', 'svm.r1cs')
        
        if not os.path.exists(r1cs_file):
            print("❌ Circuit not compiled yet. Compiling first...")
            if not compile_circuit():
                print("❌ Cannot start API without compiled circuit")
                return
        
        # Start the API
        start_api()

if __name__ == "__main__":
    main()
