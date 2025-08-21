import os
import numpy as np
import zkpy

def compile_svm_circuit(weights_path, bias_path, circuit_out_dir):
    os.makedirs(circuit_out_dir, exist_ok=True)
    weights = np.load(weights_path)
    bias = np.load(bias_path)
    n_features = weights.size
    # Define circuit in zkpy DSL
    circ = zkpy.circuit.CircuitBuilder()
    w_vars = [circ.input(f'w{i}') for i in range(n_features)]
    b_var = circ.input('b')
    x_vars = [circ.input(f'x{i}') for i in range(n_features)]
    dot = circ.add(sum([circ.mul(w, x) for w, x in zip(w_vars, x_vars)]), b_var)
    circ.output(dot)
    circ_file = os.path.join(circuit_out_dir, 'svm.circ')
    circ.compile(circ_file)
    print(f"SVM circuit compiled and saved to {circ_file}")
    return circ_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--bias', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    compile_svm_circuit(args.weights, args.bias, args.outdir)
