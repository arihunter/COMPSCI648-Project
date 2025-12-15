import torch
import math
import random

from quantum_simulator import (
    zero_state,
    apply_gate,
    RY,
    RZ,
    CNOT,
    state_to_density,
    build_full_unitary,
    kraus_operator,
    apply_named_gate_density,
    apply_T1T2_noise_op,
)

# sklearn only for data (this is standard + allowed)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ============================================================
# Dataset (REAL-WORLD)
# ============================================================
def make_real_dataset(test_size=0.3, seed=42):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # labels: {0,1} → {-1,+1}
    y = 2 * y - 1

    # normalize
    X = StandardScaler().fit_transform(X)

    # reduce to 2 features → 2 qubits
    X = PCA(n_components=2).fit_transform(X)

    # scale to [-π, π] for angle encoding
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
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    I = torch.eye(2, dtype=torch.cfloat)
    op = None
    for q in range(n):
        mat = Z if q == qubit else I
        op = mat if op is None else torch.kron(op, mat)
    return torch.real(state.conj() @ (op @ state))

# ============================================================
# Deep Variational QNN
# ============================================================
def deep_vqc_forward(x, theta, depth=3):
    n = 2
    state = zero_state(n)

    state = apply_gate(state, RY(x[0]), [0], n)
    state = apply_gate(state, RY(x[1]), [1], n)

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
def noisy_qnn_forward(x, theta, T1=100, T2=200):
    n = 2
    density = state_to_density(zero_state(n))

    for gate in [
        build_full_unitary(RY(x[0]), [0], n),
        build_full_unitary(RY(x[1]), [1], n),
        build_full_unitary(RY(theta[0]), [0], n),
        build_full_unitary(RY(theta[1]), [1], n),
    ]:
        density = kraus_operator(density, [(gate, 1.0)])

    density = apply_named_gate_density(density, ("CNOT", [0,1]), n)

    U = build_full_unitary(RZ(theta[2]), [0], n)
    density = kraus_operator(density, [(U, 1.0)])

    density = apply_T1T2_noise_op(
        density,
        ("T1T2_NOISE", [0], T1, T2, 1),
        n
    )

    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
    I = torch.eye(2, dtype=torch.cfloat)
    Z0 = torch.kron(Z, I)

    return torch.real(torch.trace(Z0 @ density))

# ============================================================
# Quantum Kernel Model
# ============================================================
def quantum_feature_map(x):
    n = 2
    state = zero_state(n)
    state = apply_gate(state, RY(x[0]), [0], n)
    state = apply_gate(state, RY(x[1]), [1], n)
    state = apply_gate(state, CNOT, [0,1], n)
    return state

def quantum_kernel(x1, x2):
    ψ1 = quantum_feature_map(x1)
    ψ2 = quantum_feature_map(x2)
    return torch.abs(torch.dot(ψ1.conj(), ψ2))**2

def kernel_predict(x, X_train, y_train):
    vals = torch.tensor([quantum_kernel(x, xi) for xi in X_train])
    return torch.sign(torch.sum(vals * y_train))

# ============================================================
# Training
# ============================================================
def train(model_type="deep_vqc"):
    X_train, X_test, y_train, y_test = make_real_dataset()
    epochs = 25
    lr = 0.1

    if model_type == "kernel":
        scores = torch.tensor([
            kernel_predict(x, X_train, y_train) for x in X_test
        ])
        metrics = compute_metrics(scores, y_test)
        print("KERNEL TEST METRICS:", metrics)
        return

    theta = torch.randn(9 if model_type == "deep_vqc" else 3) * 0.1

    for epoch in range(epochs):
        grads = torch.zeros_like(theta)

        for i in range(len(X_train)):
            xi, yi = X_train[i], y_train[i]

            pred = (
                deep_vqc_forward(xi, theta)
                if model_type == "deep_vqc"
                else noisy_qnn_forward(xi, theta)
            )

            for p in range(len(theta)):
                shift = math.pi / 2
                tp, tm = theta.clone(), theta.clone()
                tp[p] += shift
                tm[p] -= shift

                fp = (
                    deep_vqc_forward(xi, tp)
                    if model_type == "deep_vqc"
                    else noisy_qnn_forward(xi, tp)
                )
                fm = (
                    deep_vqc_forward(xi, tm)
                    if model_type == "deep_vqc"
                    else noisy_qnn_forward(xi, tm)
                )

                grads[p] += 0.5 * ((fp - yi)**2 - (fm - yi)**2)

        theta -= lr * grads / len(X_train)

        scores_test = torch.tensor([
            deep_vqc_forward(x, theta)
            if model_type == "deep_vqc"
            else noisy_qnn_forward(x, theta)
            for x in X_test
        ])

        metrics = compute_metrics(scores_test, y_test)

        print(
            f"{model_type.upper()} | Epoch {epoch:02d} | "
            f"Acc {metrics['accuracy']:.3f} | "
            f"Prec {metrics['precision']:.3f} | "
            f"Rec {metrics['recall']:.3f} | "
            f"F1 {metrics['f1']:.3f} | "
            f"AUC {metrics['roc_auc']:.3f}"
        )

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("\nTraining Deep Variational QNN (Real Dataset)")
    train("deep_vqc")

    print("\nTraining Noise-Aware QNN (Real Dataset)")
    train("noise_aware")

    print("\nTraining Quantum Kernel Model (Real Dataset)")
    train("kernel")
