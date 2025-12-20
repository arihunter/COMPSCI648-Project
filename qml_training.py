from quantum_simulator import run_noisy_circuit_density
import torch
import math
import random
import os
from enum import Enum

from constants import DEFAULT_GATE_DURATIONS, DEFAULT_T1, DEFAULT_T2

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
    """Load breast-cancer data, map labels to {-1,+1}, and return train/test splits for the chosen encoding."""
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
    """Convert real-valued scores to {-1,+1} labels."""
    return torch.where(scores >= 0, 1.0, -1.0)

def accuracy(y_true, y_pred):
    """Binary accuracy for {-1,+1} targets."""
    return (y_true == y_pred).float().mean().item()

def precision(y_true, y_pred):
    """Positive predictive value for {-1,+1} labels."""
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == -1)).sum().item()
    return tp / (tp + fp + 1e-9)

def recall(y_true, y_pred):
    """True positive rate for {-1,+1} labels."""
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred == -1) & (y_true == 1)).sum().item()
    return tp / (tp + fn + 1e-9)

def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)

def roc_auc_score(y_true, scores):
    """Manual ROC-AUC for {-1,+1} labels and real-valued scores."""
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
    """Return basic classification metrics from scores and true labels."""
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
    """Compute ⟨Z⟩ on the specified qubit for a ket state."""
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
# Deep Variational QNN, with optional T1/T2 noise
# ============================================================
def deep_vqc_forward(x, theta, depth=3, encoding=EncodingType.ANGLE, n=None, T1=None, T2=None, gate_durations=None, gate_noise_fraction=1.0, cached_encoding=None):
    """Forward pass of the deep VQC with optional T1/T2 noise, returning ⟨Z0⟩.
    
    Args:
        cached_encoding: If provided, tuple (init_state, encoding_circuit) to skip state_encoding call.
                         Allows reuse of encoding across multiple forward evaluations with different theta.
    """
    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
    
    # Check theta length matches depth
    assert len(theta) == 3*depth
    
    # Calculate number of qubits based on encoding type
    if n is None:
        n = len(x) if encoding == EncodingType.ANGLE else int(math.log2(len(x)))
    
    # Use cached encoding if provided, otherwise compute it
    if cached_encoding is not None:
        init_state, circ = cached_encoding
        # Make a copy of the circuit so we don't modify the cached version
        circ = list(circ)
    else:
        init_state, circ = state_encoding(x, n, encoding)
    
    # Add variational layers to circuit
    idx = 0
    for _ in range(depth):
        circ += [("RY", [0], theta[idx])]
        circ += [("RY", [1], theta[idx+1])]
        circ += [("CNOT", [0, 1])]
        circ += [("RZ", [0], theta[idx+2])]
        idx += 3
    
    # Use noisy density simulation if T1/T2 provided
    if T1 is not None and T2 is not None:
        density = run_noisy_circuit_density(
            initial_state=init_state,
            circuit=circ,
            num_qubits=n,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations,
            gate_noise_fraction=gate_noise_fraction
        )
        Z0 = torch.kron(Z, I2)
        return torch.real(torch.trace(Z0 @ density))
    else:
        # Noiseless ket simulation
        state = apply_circuit_to_ket(init_state, circ, n)
        return expectation_z(state, 0, n)

# ============================================================
# Noise-aware QNN (density matrix)
# ============================================================

# Helper function to build layer of RX/Y/Z on selected qubits
def param_gate_layer(gate: str, x, specify_qubits: tuple[int] | None = None):
    """Construct a list of single-qubit parameterized gates over given qubits."""
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
def noisy_qnn_forward(x, theta, T1=DEFAULT_T1, T2=DEFAULT_T2,
    gate_durations=None,  # needed for noise model (all times in μs)
    encoding=EncodingType.ANGLE,
    gate_noise_fraction=1.0,
    cached_encoding=None):  # Optional cached (init_state, encoding_circuit)

    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
    
    # Calculate number of qubits based on encoding type
    n = len(x) if encoding == EncodingType.ANGLE else int(math.log2(len(x)))
    
    # Use cached encoding if provided, otherwise compute it
    if cached_encoding is not None:
        init_state, circ = cached_encoding
        circ = list(circ)  # Copy to avoid modifying cached version
    else:
        init_state, circ = state_encoding(x, n, encoding)  # circ now has the first RY layer (if angle encoding)
    
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
        gate_durations = gate_durations,
        gate_noise_fraction = gate_noise_fraction
    )
    
    Z0 = torch.kron(Z, I2)

    return torch.real(torch.trace(Z0 @ density))

# ============================================================
# Quantum Kernel Model
# ============================================================
def quantum_feature_map(x, encoding=EncodingType.ANGLE, T1=None, T2=None,
    gate_durations=None):
    """Build the feature map circuit/state for kernel or noisy simulation."""
    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
    # Calculate number of qubits based on encoding type
    n = len(x) if encoding == EncodingType.ANGLE else int(math.log2(len(x)))
    init_state, circ = state_encoding(x, n, encoding)
    
    # Add CNOT to circuit
    circ += [("CNOT", [0, 1])]
    
    # If noise parameters provided, use noisy simulation
    if T1 is not None and T2 is not None:
        density = run_noisy_circuit_density(
            initial_state=init_state,
            circuit=circ,
            num_qubits=n,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations
        )
        return density
    else:
        # Noiseless case: apply circuit to ket vector
        state = apply_circuit_to_ket(init_state, circ, n)
        return state

def quantum_kernel(x1, x2, encoding=EncodingType.ANGLE, T1=None, T2=None,
    gate_durations=None):
    """Compute kernel value between two inputs via feature map (ket or density)."""
    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
    ψ1 = quantum_feature_map(x1, encoding, T1, T2, gate_durations)
    ψ2 = quantum_feature_map(x2, encoding, T1, T2, gate_durations)
    
    # Handle both ket vectors and density matrices
    if isinstance(ψ1, torch.Tensor) and ψ1.dim() == 1:
        # Noiseless: ket vector inner product
        return torch.abs(torch.dot(ψ1.conj(), ψ2))**2
    else:
        # Noisy: trace distance between density matrices
        return torch.real(torch.trace(ψ1 @ ψ2))

def kernel_predict_batch(X_test, X_train, y_train, encoding=EncodingType.ANGLE, T1=None, T2=None,
    gate_durations=None):
    """Batch predict using cached training feature maps - much faster than repeated kernel_predict.
    
    Parallelizes feature map computation across all available cores.
    Computes feature maps once per training/test point instead of once per pair.
    """
    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
    
    # Set up multiprocessing for feature map computation (parallelized by default)
    from torch.multiprocessing import Pool, set_start_method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    n_workers = max(1, os.cpu_count() - 1)
    pool = None
    try:
        pool = Pool(processes=n_workers)
        
        # Collect all feature map tasks
        all_tasks = []
        for xi in X_train:
            all_tasks.append((xi, encoding, T1, T2))
        for xi in X_test:
            all_tasks.append((xi, encoding, T1, T2))
        
        # Compute all feature maps in parallel
        print(f"Computing {len(X_train)} + {len(X_test)} feature maps across {n_workers} workers...")
        chunksize = max(1, len(all_tasks) // (n_workers * 4))
        features_list = list(pool.imap_unordered(_evaluate_single_feature_map_task, 
                                                  all_tasks, chunksize=chunksize))
        
        train_features = features_list[:len(X_train)]
        test_features = features_list[len(X_train):]
        
        pool.close()
        pool.join()
    except Exception as e:
        print(f"Warning: Could not set up multiprocessing ({e}), falling back to sequential")
        # Pre-compute all training feature maps sequentially
        print(f"Computing {len(X_train)} training feature maps...")
        train_features = [quantum_feature_map(xi, encoding, T1, T2, gate_durations) for xi in X_train]
        
        print(f"Computing {len(X_test)} test feature maps...")
        test_features = [quantum_feature_map(x, encoding, T1, T2, gate_durations) for x in X_test]
    
    # For each test point, compute its feature map and compare against cached training features
    predictions = []
    for i, x_feature in enumerate(test_features):
        if (i + 1) % max(1, len(X_test) // 10) == 0 or i == 0:
            print(f"  Computing kernel overlaps for test point {i+1}/{len(X_test)}...")
        
        # Compute kernel values against all training points using cached features
        if isinstance(x_feature, torch.Tensor) and x_feature.dim() == 1:
            # Noiseless: ket vector inner products
            vals = torch.tensor([torch.abs(torch.dot(x_feature.conj(), ψi))**2 for ψi in train_features])
        else:
            # Noisy: trace overlaps with density matrices
            vals = torch.tensor([torch.real(torch.trace(x_feature @ ψi)) for ψi in train_features])
        
        pred = torch.sign(torch.sum(vals * y_train))
        predictions.append(pred)
    
    return torch.tensor(predictions)

def kernel_predict(x, X_train, y_train, encoding=EncodingType.ANGLE, T1=None, T2=None,
    gate_durations=None):
    """Predict sign label via kernel sum against training set.
    
    Note: For multiple predictions, use kernel_predict_batch() which caches training features.
    """
    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
    vals = torch.tensor([
        quantum_kernel(x, xi, encoding, T1, T2, gate_durations) for xi in X_train
    ])
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
# ───────────────────────────────────────────────────────────────
# Parallelized training helpers
# ───────────────────────────────────────────────────────────────

def _evaluate_single_gradient_task(task):
    """
    Evaluate a single parameter-shift gradient task.
    
    Task format: (sample_idx, xi, yi, theta, param_idx, model_type, encoding_cache, 
                  deep_vqc_depth, T1, T2, encoding, shift_direction)
    
    Returns: (sample_idx, param_idx, shift_dir, forward_value, loss_contribution)
    """
    (sample_idx, xi, yi, theta, param_idx, model_type, cached_enc,
     deep_vqc_depth, T1, T2, encoding, shift_dir) = task
    
    # Create shifted parameters
    theta_shifted = theta.clone()
    shift = math.pi / 2
    theta_shifted[param_idx] += shift * shift_dir  # +shift or -shift
    
    # Evaluate forward pass
    if model_type == "deep_vqc":
        pred = deep_vqc_forward(xi, theta_shifted, deep_vqc_depth, encoding, 
                               T1=T1, T2=T2, cached_encoding=cached_enc)
    else:  # noise_aware
        pred = noisy_qnn_forward(xi, theta_shifted, T1=T1, T2=T2, 
                                encoding=encoding, cached_encoding=cached_enc)
    
    return (sample_idx, param_idx, shift_dir, pred.item(), (pred - yi).item())


def _evaluate_sample_batch(batch_task):
    """
    Evaluate gradient contributions for a batch of training samples.
    
    This batches multiple samples together to amortize IPC costs.
    Each worker processes all parameter shifts for its assigned samples.
    
    Args:
        batch_task: (sample_indices, sample_data, theta, config)
            - sample_indices: list of sample indices
            - sample_data: list of (xi, yi, cached_enc) tuples
            - theta: parameter tensor
            - config: (model_type, deep_vqc_depth, T1, T2, encoding)
    
    Returns:
        List of (sample_idx, gradients_array, baseline_pred, loss_contrib) tuples
    """
    sample_indices, sample_data, theta, config = batch_task
    model_type, deep_vqc_depth, T1, T2, encoding = config
    
    results = []
    n_params = len(theta)
    shift = math.pi / 2
    
    for idx, (xi, yi, cached_enc) in zip(sample_indices, sample_data):
        # Compute baseline prediction
        if model_type == "deep_vqc":
            baseline_pred = deep_vqc_forward(xi, theta, deep_vqc_depth, encoding,
                                            T1=T1, T2=T2, cached_encoding=cached_enc)
        else:  # noise_aware
            baseline_pred = noisy_qnn_forward(xi, theta, T1=T1, T2=T2,
                                             encoding=encoding, cached_encoding=cached_enc)
        
        baseline_val = baseline_pred.item()
        loss_contrib = (baseline_val - yi) ** 2
        
        # Compute parameter-shift gradients for all parameters
        grads = torch.zeros(n_params)
        for p in range(n_params):
            # Positive shift
            theta_plus = theta.clone()
            theta_plus[p] += shift
            if model_type == "deep_vqc":
                pred_plus = deep_vqc_forward(xi, theta_plus, deep_vqc_depth, encoding,
                                            T1=T1, T2=T2, cached_encoding=cached_enc)
            else:
                pred_plus = noisy_qnn_forward(xi, theta_plus, T1=T1, T2=T2,
                                             encoding=encoding, cached_encoding=cached_enc)
            
            # Negative shift
            theta_minus = theta.clone()
            theta_minus[p] -= shift
            if model_type == "deep_vqc":
                pred_minus = deep_vqc_forward(xi, theta_minus, deep_vqc_depth, encoding,
                                             T1=T1, T2=T2, cached_encoding=cached_enc)
            else:
                pred_minus = noisy_qnn_forward(xi, theta_minus, T1=T1, T2=T2,
                                              encoding=encoding, cached_encoding=cached_enc)
            
            # Parameter-shift rule
            loss_plus = (pred_plus.item() - yi) ** 2
            loss_minus = (pred_minus.item() - yi) ** 2
            grads[p] = 0.5 * (loss_plus - loss_minus)
        
        results.append((idx, grads, baseline_val, loss_contrib))
    
    return results


def _evaluate_single_feature_map_task(task):
    """Evaluate a single feature map for kernel model."""
    xi, encoding, T1, T2 = task
    return quantum_feature_map(xi, encoding, T1, T2)


def train(model_type="deep_vqc", encoding=EncodingType.ANGLE, epochs=25, dataset="real", 
          record_metrics=False, T1=DEFAULT_T1, T2=DEFAULT_T2):
    if dataset == "real":
        X_train, X_test, y_train, y_test = make_real_dataset(encoding=encoding)
    elif dataset == "moons":
        X_train, X_test, y_train, y_test = make_moons_dataset(encoding=encoding)
    else:
        raise Exception("Unknown dataset")

    lr = 0.1
    deep_vqc_depth = 3

    if model_type == "kernel":
        # Use batch prediction with cached feature maps for massive speedup
        scores = kernel_predict_batch(X_test, X_train, y_train, encoding, T1, T2)
        metrics = compute_metrics(scores, y_test)
        print(f"{dataset.upper()} KERNEL TEST METRICS:", metrics)
        return

    theta = torch.randn(9 if model_type == "deep_vqc" else 3) * 0.1

    # For plotting
    loss_history = []
    acc_history = []

    # Pre-compute encodings for all training samples to reuse across epochs and parameter shifts
    n_qubits = len(X_train[0]) if encoding == EncodingType.ANGLE else int(math.log2(len(X_train[0])))
    encoding_cache = {}
    for i in range(len(X_train)):
        xi = X_train[i]
        init_state, circ = state_encoding(xi, n_qubits, encoding)
        encoding_cache[i] = (init_state, circ)

    # Set up multiprocessing for gradient evaluation (parallelized by default)
    from torch.multiprocessing import Pool, set_start_method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    n_workers = max(1, os.cpu_count() - 1)
    pool = None
    try:
        pool = Pool(processes=n_workers)
    except Exception as e:
        print(f"Warning: Could not set up multiprocessing ({e}), falling back to sequential")

    for epoch in range(epochs):
        grads = torch.zeros_like(theta)
        total_loss = 0.0

        if pool is not None:
            # Create batched tasks: group samples to reduce IPC overhead
            # Each batch processes multiple samples together in one worker call
            batch_size = max(5, len(X_train) // (n_workers * 2))  # 2 batches per worker minimum
            batch_tasks = []
            
            # Prepare shared config to avoid repeated serialization
            config = (model_type, deep_vqc_depth, T1, T2, encoding)
            
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                sample_indices = list(range(batch_start, batch_end))
                sample_data = [(X_train[i], y_train[i], encoding_cache[i]) 
                              for i in sample_indices]
                
                batch_tasks.append((sample_indices, sample_data, theta, config))
            
            # Process batches in parallel with larger chunks
            # Target: each chunk should have substantial computation (multiple samples)
            chunksize = max(1, len(batch_tasks) // n_workers)  # At least 1 batch per worker
            batch_results = pool.map(_evaluate_sample_batch, batch_tasks, chunksize=chunksize)
            
            # Flatten results from all batches
            results = [item for batch in batch_results for item in batch]
            
            # Process batched results
            for result in results:
                sample_idx, sample_grads, baseline_val, loss_contrib = result
                grads += sample_grads
                total_loss += loss_contrib
        else:
            # Sequential fallback
            for i in range(len(X_train)):
                xi, yi = X_train[i], y_train[i]
                cached_enc = encoding_cache[i]

                pred = (
                    deep_vqc_forward(xi, theta, deep_vqc_depth, encoding, T1=T1, T2=T2, cached_encoding=cached_enc)
                    if model_type == "deep_vqc"
                    else noisy_qnn_forward(xi, theta, T1=T1, T2=T2, encoding=encoding, cached_encoding=cached_enc)
                )

                total_loss += (pred - yi)**2

                for p in range(len(theta)):
                    shift = math.pi / 2
                    tp, tm = theta.clone(), theta.clone()
                    tp[p] += shift
                    tm[p] -= shift

                    fp = (
                        deep_vqc_forward(xi, tp, deep_vqc_depth, encoding, T1=T1, T2=T2, cached_encoding=cached_enc)
                        if model_type == "deep_vqc"
                        else noisy_qnn_forward(xi, tp, T1=T1, T2=T2, encoding=encoding, cached_encoding=cached_enc)
                    )
                    fm = (
                        deep_vqc_forward(xi, tm, deep_vqc_depth, encoding, T1=T1, T2=T2, cached_encoding=cached_enc)
                        if model_type == "deep_vqc"
                        else noisy_qnn_forward(xi, tm, T1=T1, T2=T2, encoding=encoding, cached_encoding=cached_enc)
                    )

                    grads[p] += 0.5 * ((fp - yi)**2 - (fm - yi)**2)

        # Update
        theta -= lr * grads / len(X_train)

        # Record loss
        loss_avg = (total_loss.item() if torch.is_tensor(total_loss) else total_loss) / len(X_train)
        loss_history.append(loss_avg)

        # Cache test encodings on first epoch
        if epoch == 0:
            test_encoding_cache = {}
            for j in range(len(X_test)):
                xj = X_test[j]
                init_state, circ = state_encoding(xj, n_qubits, encoding)
                test_encoding_cache[j] = (init_state, circ)

        # Test accuracy using cached test encodings
        scores_test = torch.tensor([
            deep_vqc_forward(x, theta, deep_vqc_depth, encoding, T1=T1, T2=T2, cached_encoding=test_encoding_cache[j])
            if model_type == "deep_vqc"
            else noisy_qnn_forward(x, theta, T1=T1, T2=T2, encoding=encoding, cached_encoding=test_encoding_cache[j])
            for j, x in enumerate(X_test)
        ])
        metric_dict = compute_metrics(scores_test, y_test)
        acc_history.append(metric_dict["accuracy"])

        print(
            f"{dataset.upper()} | {model_type.upper()} | Epoch {epoch+1:02d} | "
            f"Loss {loss_avg:.4f} | Acc {metric_dict['accuracy']:.3f}"
        )

    # Clean up pool
    if pool is not None:
        pool.close()
        pool.join()

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
