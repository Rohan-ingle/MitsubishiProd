"""
ZKP Environment Setup Script

This script will install the necessary dependencies for the Zero-Knowledge Proof (ZKP) system.
"""
import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Run a command and print the output"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def setup_zkp_environment():
    """Set up the ZKP environment"""
    print("Setting up ZKP environment...")
    
    # Install pysnark and its dependencies
    packages = [
        "pip install --upgrade pip",
        "pip install numpy",
        "pip install pandas",
        "pip install pysnark",
        "pip install zkpy"
    ]
    
    for package_cmd in packages:
        cmd = package_cmd.split()
        success = run_command(cmd, f"Installing {cmd[-1]}")
        if not success:
            print(f"❌ Failed to install {cmd[-1]}. Continuing with next package...")
    
    # Create necessary directories
    dirs = [
        os.path.join('models', 'zkp'),
        os.path.join('models', 'zkl')
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Created directory {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")

def setup_qaptools():
    """Set up qaptools"""
    print("\nSetting up qaptools...")
    
    system = platform.system()
    if system == "Windows":
        print("Windows system detected. Installing qaptools for Windows...")
        # For Windows, we might need to get pre-compiled binaries or setup WSL
        print("❌ qaptools direct installation on Windows is not available.")
        print("  Please consider one of these options:")
        print("  1. Use WSL (Windows Subsystem for Linux) and install qaptools there")
        print("  2. Use a Docker container with qaptools installed")
        print("  3. Modify the code to use an alternative backend that doesn't require qaptools")
    elif system == "Linux":
        print("Linux system detected. Installing qaptools for Linux...")
        # For Linux, we can try to compile qaptools
        cmd = [
            "git", "clone", "https://github.com/Charterhouse/qaptools.git", 
            "qaptools_build"
        ]
        if run_command(cmd, "Cloning qaptools repository"):
            os.chdir("qaptools_build")
            if run_command(["make"], "Building qaptools"):
                print("✓ qaptools built successfully")
                # TODO: Add the qaptools directory to PATH
                os.chdir("..")
            else:
                print("❌ Failed to build qaptools")
                os.chdir("..")
    else:
        print(f"❌ Unsupported system: {system}")

def modify_code_for_zkp_fallback():
    """Modify the code to properly handle ZKP fallback"""
    print("\nModifying code for better ZKP fallback...")
    
    # Path to the file we want to modify
    file_path = 'inferencing/bank1_classify_and_send.py'
    
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found")
        return
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if zkpy import is wrapped in try-except
    if 'try:\n    import zkpy' not in content:
        # Modify the import section to handle missing zkpy
        import_section = 'import numpy as np\nimport pandas as pd\nimport json\nimport zkpy\nfrom modules.svm_zkp import convert_model_to_zkp'
        new_import_section = '''import numpy as np
import pandas as pd
import json

# Try to import zkpy and pysnark, but handle their absence gracefully
try:
    import zkpy
    ZKPY_AVAILABLE = True
except ImportError:
    print("Warning: zkpy module not found. Will use simple inference without ZKP.")
    ZKPY_AVAILABLE = False

try:
    from modules.svm_zkp import convert_model_to_zkp
    PYSNARK_AVAILABLE = True
except ImportError:
    print("Warning: pysnark module not found. Will use simple inference without ZKP.")
    PYSNARK_AVAILABLE = False
    
    # Define a fallback function for convert_model_to_zkp
    def convert_model_to_zkp(weights_path, bias_path, metadata_out_path):
        """Fallback function when pysnark is not available"""
        import numpy as np
        import json
        import os
        
        # Load the model
        weights = np.load(weights_path)
        bias = np.load(bias_path)
        
        # Create basic metadata
        model_info = {
            'num_features': weights.shape[1],
            'weights_shape': weights.shape,
            'bias_shape': bias.shape,
            'note': 'Created without pysnark - no commitments available'
        }
        
        # Save metadata
        out_dir = os.path.dirname(metadata_out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(metadata_out_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Basic SVM metadata (without ZKP) has been created and saved to {metadata_out_path}.")
        return weights, bias'''
        
        content = content.replace(import_section, new_import_section)
    
    # Replace the ZKP circuit checking logic to handle missing zkpy more gracefully
    circuit_check = '''if os.path.exists(root_r1cs_file) and os.path.exists(root_wasm_file):
    try:
        # Use zkpy Circuit with the root files
        circuit = zkpy.Circuit(root_r1cs_file)'''
        
    new_circuit_check = '''if os.path.exists(root_r1cs_file) and os.path.exists(root_wasm_file) and ZKPY_AVAILABLE:
    try:
        # Use zkpy Circuit with the root files
        circuit = zkpy.Circuit(root_r1cs_file)'''
        
    content = content.replace(circuit_check, new_circuit_check)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Modified {file_path} to better handle missing ZKP dependencies")

if __name__ == "__main__":
    print("ZKP Setup Script")
    print("===============")
    
    # Ask for confirmation
    confirm = input("This script will install ZKP dependencies and modify code. Continue? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    setup_zkp_environment()
    setup_qaptools()
    modify_code_for_zkp_fallback()
    
    print("\nSetup complete. Please run check_zkp_environment.py to verify the installation.")
