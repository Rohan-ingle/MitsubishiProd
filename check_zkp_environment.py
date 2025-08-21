import os
import sys
import importlib

# Get the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check_module_imports():
    """Test imports of the required modules"""
    print("Checking module imports...")
    
    # Try to import zkpy
    try:
        import zkpy
        print(f"✓ zkpy module found")
        print(f"  Available attributes: {dir(zkpy)}")
        
        # Check for Circuit class
        if hasattr(zkpy, 'Circuit'):
            print(f"✓ zkpy.Circuit class found")
        else:
            print(f"❌ zkpy.Circuit class not found")
        
        # Check for scheme constants
        if hasattr(zkpy, 'GROTH'):
            print(f"✓ zkpy.GROTH scheme found")
        else:
            print(f"❌ zkpy.GROTH scheme not found")
            
    except ImportError as e:
        print(f"❌ Error importing zkpy: {e}")
    
    # Try to import pysnark
    try:
        import pysnark
        print(f"✓ pysnark module found")
        print(f"  Available attributes: {dir(pysnark)}")
        
        # Check backends
        backends = [
            'pysnark.libsnark.backend',
            'pysnark.libsnark.backendgg',
            'pysnark.qaptools.backend'
        ]
        
        for backend in backends:
            try:
                module = importlib.import_module(backend)
                print(f"✓ {backend} loaded successfully")
            except ImportError as e:
                print(f"❌ Error loading {backend}: {e}")
            except Exception as e:
                print(f"❌ Error with {backend}: {e}")
        
    except ImportError as e:
        print(f"❌ Error importing pysnark: {e}")

def check_circuit_files():
    """Check if circuit files exist"""
    print("\nChecking circuit files...")
    
    circuit_files = [
        os.path.join(PROJECT_ROOT, 'svm.circ.r1cs'),
        os.path.join(PROJECT_ROOT, 'svm.circ.wasm'),
        os.path.join(PROJECT_ROOT, 'svm.circ.sym')
    ]
    
    for file_path in circuit_files:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)} found")
        else:
            print(f"❌ {os.path.basename(file_path)} not found")

def check_qaptools():
    """Check if qaptools are available"""
    print("\nChecking qaptools executables...")
    
    import subprocess
    from shutil import which
    
    qaptools = ['qapgen', 'qapinput', 'qapcoeff', 'qapsolve', 'qapver']
    
    for tool in qaptools:
        if which(tool) or which(f"{tool}.exe"):
            print(f"✓ {tool} executable found in PATH")
        else:
            print(f"❌ {tool} executable not found in PATH")
    
    # Try to run qapgen to check if it's available
    try:
        result = subprocess.run(['qapgen', '--help'], 
                               capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"✓ qapgen executed successfully")
        else:
            print(f"❌ qapgen execution failed with return code {result.returncode}")
            print(f"  Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running qapgen: {e}")

def check_zkp_dirs():
    """Check if ZKP directories exist"""
    print("\nChecking ZKP directories...")
    
    zkp_dirs = [
        os.path.join(PROJECT_ROOT, 'models', 'zkp'),
        os.path.join(PROJECT_ROOT, 'models', 'zkl')
    ]
    
    for directory in zkp_dirs:
        if os.path.exists(directory):
            print(f"✓ {os.path.relpath(directory, PROJECT_ROOT)} exists")
        else:
            print(f"❌ {os.path.relpath(directory, PROJECT_ROOT)} does not exist")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"  Created {os.path.relpath(directory, PROJECT_ROOT)}")
            except Exception as e:
                print(f"  Error creating directory: {e}")

if __name__ == "__main__":
    print("ZKP Environment Checker")
    print("======================")
    
    check_module_imports()
    check_circuit_files()
    check_qaptools()
    check_zkp_dirs()
    
    print("\nDiagnosis complete.")
