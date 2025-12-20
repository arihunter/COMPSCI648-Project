from quantum_simulator import run_noisy_circuit_density
import torch
import math
import random
from enum import Enum

DEBUG = False

class EncodingType(Enum):
    """Enumeration for quantum feature encoding types."""
    ANGLE = "angle"
    AMPLITUDE = "amplitude"
    
from quantum_simulator import (
    zero_state,
    custom_state,
    apply_gate,
    apply_circuit_to_ket,
    RY,
    RZ,
    RX,
    CNOT,
    Z,
    I2
)

# sklearn only for data (this is standard + allowed)
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ============================================================
# Dataset (REAL-WORLD)
# ============================================================
def make_real_dataset(test_size=0.3, seed=42, encoding=EncodingType.ANGLE):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # labels: {0,1} → {-1,+1}
    y = 2 * y - 1

    # normalize
    X = StandardScaler().fit_transform(X)

    # For angle encoding: n features → n qubits (use 2 features for 2 qubits)
    # For amplitude encoding: 2^n amplitudes needed for n qubits (use 4 features for 2 qubits)
    n_components = 4 if encoding == EncodingType.AMPLITUDE else 2
    X = PCA(n_components=n_components).fit_transform(X)

    # scale to [-π, π] for angle encoding (or normalized range for amplitude)
    if encoding == EncodingType.ANGLE:
        X = math.pi * X / X.max(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

# ============================================================
# New: make_moons Dataset
# ============================================================

def make_moons_dataset(test_size=0.3, seed=42, noise=0.15, encoding=EncodingType.ANGLE):
    """
    Generates a binary classification dataset (two interleaving moons).
    Labels {0,1} are mapped to {-1,+1}.
    """
    X, y = make_moons(n_samples=500, noise=noise, random_state=seed)  # binary labels 0/1  
    y = 2 * y - 1  # convert {0,1} to {-1,+1}

    # Standardize base 2D features
    X = StandardScaler().fit_transform(X)

    # Match feature dimensionality to encoding logic (PCA where possible)
    # - ANGLE encoding: 2 components → 2 qubits
    # - AMPLITUDE encoding: need 4 real features → 2^2 amplitudes for 2 qubits
    if encoding == EncodingType.AMPLITUDE:
        if X.shape[1] < 4:
            # Pad with zeros in alternating slots: [x1, 0, x2, 0]
            # Each adjacent pair feeds one qubit amplitude; keeping one value non-zero per pair
            # avoids initializing both amplitudes to zero and wasting that qubit.
            X_pad = torch.zeros((X.shape[0], 4), dtype=torch.float32)
            X_pad[:, 0] = torch.tensor(X[:, 0], dtype=torch.float32)
            X_pad[:, 2] = torch.tensor(X[:, 1], dtype=torch.float32)
            X = X_pad.numpy()
        else:
            X = PCA(n_components=4).fit_transform(X)
    else:
        X = PCA(n_components=2).fit_transform(X)

    # Scale to [-π, π] for ANGLE encoding (consistent with make_real_dataset)
    if encoding == EncodingType.ANGLE:
        X = math.pi * X / X.max(axis=0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


# ============================================================
# Metrics
# ============================================================
def binary_predictions(scores):
    return torch.where(scores >= 0, 1.0, -1.0)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()

def precision(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == -1)).sum().item()
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred == -1) & (y_true == 1)).sum().item()
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)

def roc_auc_score(y_true, scores):
    idx = torch.argsort(scores, descending=True)
    y = y_true[idx]

    pos = (y == 1).sum().item()
    neg = (y == -1).sum().item()
    if pos == 0 or neg == 0:
        return 0.0

    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    prev_fpr = 0.0

    for label in y:
        if label == 1:
            tpr += 1 / pos
        else:
            fpr += 1 / neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return auc

def compute_metrics(scores, y_true):
    y_pred = binary_predictions(scores)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
    }

# ============================================================
# Quantum utilities
# ============================================================
def expectation_z(state, qubit, n):
    op = None
    for q in range(n):
        mat = Z if q == qubit else I2
        op = mat if op is None else torch.kron(op, mat)
    return torch.real(state.conj() @ (op @ state))

    """
    Encoding methods - angle encoding, and amplitude encoding. Returns normalized ket vector state, and circuit

    x - input vector
    n - amount of qubits
    encoding - enum EncodingType

    Throws error if the encoding type is not known.
    Will return state and circ, if circ is nonempty - the state encoding method needs it to be applied to the state to be initialized. This allows for error models that apply noise during state preparation.

    EX:
    state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
    TO GET KET:
    state = apply_circuit_to_ket(state, circuit, n)

    TO USE DENSITY SIM WITH NOISE:
    density = state_to_density(state)
    density = run_noisy_circuit_density(density, circuit, n, T1, T2, gate_durations)
    """
def state_encoding(x, n, encoding=EncodingType.ANGLE):
    # Specify encoding type 
    # custom_state(x) -> Creates arbitrary normalized amplitude state (ignores gate errors, just creats perfect state)
    if encoding == EncodingType.ANGLE:
        assert len(x) == n, "Input dimension must match number of qubits for angle encoding"
        state = zero_state(n)
    elif encoding == EncodingType.AMPLITUDE:
        # For amplitude encoding, we need 2^n amplitudes
        # Pad the input if necessary
        required_size = 2 ** n
        if len(x) < required_size:
            print("Warning: Input size less than required for amplitude encoding, padding with zeros.")
            # Pad with zeros to match required size
            padded_x = torch.zeros(required_size)
            if isinstance(x, torch.Tensor):
                padded_x[:len(x)] = x.clone().detach()
            else:
                padded_x[:len(x)] = torch.tensor(x)
            state = custom_state(padded_x)
        else:
            # Use first 2^n elements if input is larger
            state = custom_state(x[:required_size])
    else:
        raise Exception("Unknown encoding")
    
    circuit = []
    # Angle encoding state, assuming x is mapped to [-π,π]
    if encoding == EncodingType.ANGLE:
        circuit = param_gate_layer("RY",x)

    return state, circuit

# ============================================================
# Deep Variational QNN, no errors
# ============================================================
def deep_vqc_forward(x, theta, depth=3, encoding=EncodingType.ANGLE, n=None):
    # Calculate number of qubits based on encoding type
    if n is None:
        n = len(x) if encoding == EncodingType.ANGLE else int(math.log2(len(x)))
    
    # Prep state with desired encoding method
    state,circ = state_encoding(x,n,encoding)
    state = apply_circuit_to_ket(state,circ,n)
    
    # Variational layers
    idx = 0
    for _ in range(depth):
        state = apply_gate(state, RY(theta[idx]), [0], n)
        state = apply_gate(state, RY(theta[idx+1]), [1], n)
        state = apply_gate(state, CNOT, [0,1], n)
        state = apply_gate(state, RZ(theta[idx+2]), [0], n)
        idx += 3

    return expectation_z(state, 0, n)

# ============================================================
# Noise-aware QNN (density matrix)
# ============================================================

# Helper function to build layer of RX/Y/Z on selected qubits
def param_gate_layer(gate: str, x, specify_qubits: tuple[int] | None = None):
    op_list = []
    if specify_qubits == None:
        # Apply gate to all qubits
        n = len(x)
        for qubit in range(n):
            op_list.append((gate, [qubit], x[qubit]))
    else:
        # apply gate to specified qubits
        for qubit in specify_qubits:
            op_list.append((gate, [qubit], x[qubit]))
    return op_list

"""
Noisy-VQC forward pass (one layer), with T1/T2 noise model.
"""
def noisy_qnn_forward(x, theta, T1=100, T2=200,
    gate_durations={ # needed for noise model
        "CNOT": 1,
        "RY"  :  1, 
    },
    encoding=EncodingType.ANGLE):
    
    # Calculate number of qubits based on encoding type
    n = len(x) if encoding == EncodingType.ANGLE else int(math.log2(len(x)))
    
    init_state, circ = state_encoding(x,n,encoding) # circ now has the first RY layer (if angle encoding)
    
    # First RY trainable layer
    circ += param_gate_layer("RY",theta,[0,1]) 
        
    # CNOT layer (0 -> 1)
    circ += [("CNOT", [0,1])]
    
    # Last trainable RZ on qubit 0
    circ += [("RZ", [0], theta[2])]
    
    if DEBUG:
        print(f"QNN layer gates:\n{circ}")
        
    density = run_noisy_circuit_density(
        initial_state=init_state,
        circuit=circ,
        num_qubits = n,
        T1 = T1,
        T2 = T2,
        gate_durations = gate_durations
    )
    
    Z0 = torch.kron(Z, I2)

    return torch.real(torch.trace(Z0 @ density))

# ============================================================
# Quantum Kernel Model
# ============================================================

def quantum_feature_map(x, encoding=EncodingType.ANGLE):
    # Calculate number of qubits based on encoding type
    n = len(x) if encoding == EncodingType.ANGLE else int(math.log2(len(x)))
    init_state,circ = state_encoding(x,n,encoding)
    state = apply_circuit_to_ket(init_state,circ,n)
    state = apply_gate(state, CNOT, [0,1], n)
    return state

def quantum_kernel(x1, x2, encoding = EncodingType.ANGLE):
    ψ1 = quantum_feature_map(x1, encoding)
    ψ2 = quantum_feature_map(x2, encoding)
    return torch.abs(torch.dot(ψ1.conj(), ψ2))**2

def kernel_predict(x, X_train, y_train, encoding = EncodingType.ANGLE):
    vals = torch.tensor([quantum_kernel(x, xi, encoding) for xi in X_train])
    return torch.sign(torch.sum(vals * y_train))

# ============================================================
# Training
# ============================================================
    """Run model training
    
    model_type: 
        "deep_vqc" - variational quantum circuit with 3 layers
        "noise_aware" - noise-aware QNN with 1 layer
        "kernel" - quantum kernel model (no training, just prediction)
    encoding:
        EncodingType.ANGLE - angle encoding
        EncodingType.AMPLITUDE - amplitude encoding
    epochs: number of training epochs
    dataset:
        "real" - breast cancer dataset
        "moons" - make_moons dataset
    record_metrics: if True, returns training metrics (loss, accuracy) history for plotting
    T1: T1 relaxation time for noise model (µs)
    T2: T2 dephasing time for noise model (µs)
    
    Returns:
        If record_metrics=True: dict with loss_history, acc_history, final_metrics, etc.
        Otherwise: None

    """
def train(model_type="deep_vqc", encoding=EncodingType.ANGLE, epochs=25, dataset="real", record_metrics=False, T1=100, T2=200):
    if dataset == "real":
        X_train, X_test, y_train, y_test = make_real_dataset(encoding=encoding)
    elif dataset == "moons":
        X_train, X_test, y_train, y_test = make_moons_dataset(encoding=encoding)
    else:
        raise Exception("Unknown dataset")

    lr = 0.1
    deep_vqc_depth = 3

    if model_type == "kernel":
        scores = torch.tensor([
            kernel_predict(x, X_train, y_train, encoding) for x in X_test
        ])
        metrics = compute_metrics(scores, y_test)
        print(f"{dataset.upper()} KERNEL TEST METRICS:", metrics)
        return

    theta = torch.randn(9 if model_type == "deep_vqc" else 3) * 0.1

    # For plotting
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        grads = torch.zeros_like(theta)
        total_loss = 0.0

        for i in range(len(X_train)):
            xi, yi = X_train[i], y_train[i]

            pred = (
                deep_vqc_forward(xi, theta, deep_vqc_depth, encoding)
                if model_type == "deep_vqc"
                else noisy_qnn_forward(xi, theta, T1=T1, T2=T2, encoding=encoding)
            )

            total_loss += (pred - yi)**2

            for p in range(len(theta)):
                shift = math.pi / 2
                tp, tm = theta.clone(), theta.clone()
                tp[p] += shift
                tm[p] -= shift

                fp = (
                    deep_vqc_forward(xi, tp, deep_vqc_depth, encoding)
                    if model_type == "deep_vqc"
                    else noisy_qnn_forward(xi, tp, T1=T1, T2=T2, encoding=encoding)
                )
                fm = (
                    deep_vqc_forward(xi, tm, deep_vqc_depth, encoding)
                    if model_type == "deep_vqc"
                    else noisy_qnn_forward(xi, tm, T1=T1, T2=T2, encoding=encoding)
                )

                grads[p] += 0.5 * ((fp - yi)**2 - (fm - yi)**2)

        # Update
        theta -= lr * grads / len(X_train)

        # Record loss
        loss_avg = total_loss.item() / len(X_train)
        loss_history.append(loss_avg)

        # Test accuracy
        scores_test = torch.tensor([
            deep_vqc_forward(x, theta, deep_vqc_depth, encoding)
            if model_type == "deep_vqc"
            else noisy_qnn_forward(x, theta, T1=T1, T2=T2, encoding=encoding)
            for x in X_test
        ])
        metric_dict = compute_metrics(scores_test, y_test)
        acc_history.append(metric_dict["accuracy"])

        print(
            f"{dataset.upper()} | {model_type.upper()} | Epoch {epoch+1:02d} | "
            f"Loss {loss_avg:.4f} | Acc {metric_dict['accuracy']:.3f}"
        )

    # Return metrics history if requested
    if record_metrics:
        return {
            'loss_history': loss_history,
            'acc_history': acc_history,
            'final_metrics': metric_dict,
            'model_type': model_type,
            'dataset': dataset,
            'encoding': encoding.value,
            'epochs': epochs
        }



# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    # Test angle encoding
    print("Training Deep Variational QNN (Angle Encoding, Real Dataset)")
    train("deep_vqc",EncodingType.ANGLE)
    print("\nTraining Noise-Aware QNN (Angle Encoding, Real Dataset)")
    train("noise_aware",EncodingType.ANGLE)
    print("\nTraining Quantum Kernel Model (Angle Encoding, Real Dataset)")
    train("kernel",EncodingType.ANGLE)
    
    # Amplitude encoding tests (may be slow)
    print("\nTraining Deep Variational QNN (Amplitude Encoding, Real Dataset)")
    train("deep_vqc",EncodingType.AMPLITUDE)
    print("\nTraining Noise-Aware QNN (Amplitude Encoding, Real Dataset)")
    train("noise_aware",EncodingType.AMPLITUDE)
    print("\nTraining Quantum Kernel Model (Amplitude Encoding, Real Dataset)")
    train("kernel",EncodingType.AMPLITUDE)  
    
    
    # print("\nTraining Deep Variational QNN (Real Dataset)")
    # train("deep_vqc")

    # print("\nTraining Noise-Aware QNN (Real Dataset)")
    # train("noise_aware")

    # print("\nTraining Quantum Kernel Model (Real Dataset)")
    # train("kernel")
