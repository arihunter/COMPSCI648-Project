"""
Parallelized QML Training
--------------------------
This module implements parallel versions of QML training functions
while keeping the core quantum circuit logic intact.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import math
from functools import partial

# Import all core functions from original module
from qml_training import (
    make_real_dataset,
    compute_metrics,
    deep_vqc_forward,
    noisy_qnn_forward,
    kernel_predict
)


# ============================================================
# Parallel Gradient Computation
# ============================================================
def compute_single_param_gradient(args):
    """
    Worker function to compute gradient for a single parameter.
    Uses parameter shift rule: ∇f = (f(θ+π/2) - f(θ-π/2)) / 2
    
    Args:
        args: Tuple of (xi, yi, theta, param_idx, model_type)
    
    Returns:
        Gradient contribution for the specified parameter
    """
    xi, yi, theta, param_idx, model_type = args
    shift = math.pi / 2
    
    # Create shifted parameter vectors
    tp = theta.clone()
    tm = theta.clone()
    tp[param_idx] += shift
    tm[param_idx] -= shift
    
    # Evaluate circuit at shifted parameters
    if model_type == "deep_vqc":
        fp = deep_vqc_forward(xi, tp)
        fm = deep_vqc_forward(xi, tm)
    else:  # noise_aware
        fp = noisy_qnn_forward(xi, tp)
        fm = noisy_qnn_forward(xi, tm)
    
    # Compute gradient of squared loss
    grad = 0.5 * ((fp - yi)**2 - (fm - yi)**2)
    return grad.item() if torch.is_tensor(grad) else grad


def parallel_gradient_step(X_train, y_train, theta, model_type, num_workers=4):
    """
    Compute gradients for all parameters in parallel across all samples.
    
    Strategy:
    - Batch all tasks: (sample_idx, param_idx, theta, model_type)
    - Single ProcessPoolExecutor for all gradient computations
    - More efficient than opening/closing executor per sample
    
    Args:
        X_train: Training features
        y_train: Training labels
        theta: Current parameters
        model_type: "deep_vqc" or "noise_aware"
        num_workers: Number of parallel workers
    
    Returns:
        Accumulated gradient vector
    """
    grads = torch.zeros_like(theta)
    
    # Create all tasks at once (sample_idx, param_idx pairs)
    tasks = []
    for i in range(len(X_train)):
        for p in range(len(theta)):
            tasks.append((X_train[i], y_train[i], theta, p, model_type))
    
    # Compute all gradients in one batch
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_grads = list(executor.map(compute_single_param_gradient, tasks))
    
    # Reshape results back and accumulate
    task_idx = 0
    for i in range(len(X_train)):
        for p in range(len(theta)):
            grads[p] += all_grads[task_idx]
            task_idx += 1
    
    return grads


# ============================================================
# Parallel Forward Pass (Batch Processing)
# ============================================================
def parallel_forward_pass(X_batch, theta, model_type, num_workers=4):
    """
    Evaluate quantum circuit on multiple samples in parallel.
    
    Args:
        X_batch: Batch of input features [batch_size, n_features]
        theta: Circuit parameters
        model_type: "deep_vqc" or "noise_aware"
        num_workers: Number of parallel workers
    
    Returns:
        Tensor of predictions for each sample
    """
    def forward_single(x):
        if model_type == "deep_vqc":
            return deep_vqc_forward(x, theta)
        else:
            return noisy_qnn_forward(x, theta)
    
    # Use thread pool for I/O bound quantum simulations
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(forward_single, X_batch))
    
    return torch.tensor(results)


# ============================================================
# Parallel Training Loop
# ============================================================
def train_parallel(model_type="deep_vqc", num_workers=4, parallel_mode="gradient"):
    """
    Train QNN with parallelization
    
    Args:
        model_type: "deep_vqc" or "noise_aware"
        num_workers: Number of parallel workers
        parallel_mode: "gradient" (parallelize gradient computation) or 
                      "batch" (parallelize forward passes)
    """
    X_train, X_test, y_train, y_test = make_real_dataset()
    epochs = 25  # Reduced from 25
    # Higher learning rate for deep_vqc to escape barren plateaus
    lr = 0.1
    
    
    theta = torch.randn(9 if model_type == "deep_vqc" else 3) * 0.1
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"Mode: {parallel_mode.upper()} parallelization | Workers: {num_workers}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        if parallel_mode == "gradient":
            # Parallel gradient computation
            grads = parallel_gradient_step(
                X_train, y_train, theta, model_type, num_workers
            )
        else:
            # Sequential gradient computation (but parallel forward passes)
            grads = torch.zeros_like(theta)
            
            for i in range(len(X_train)):
                xi, yi = X_train[i], y_train[i]
                
                for p in range(len(theta)):
                    shift = math.pi / 2
                    tp, tm = theta.clone(), theta.clone()
                    tp[p] += shift
                    tm[p] -= shift
                    
                    # Parallel forward pass
                    results = []
                    for t in [tp, tm]:
                        if model_type == "deep_vqc":
                            results.append(deep_vqc_forward(xi, t))
                        else:
                            results.append(noisy_qnn_forward(xi, t))
                    
                    fp, fm = results[0], results[1]
                    grads[p] += 0.5 * ((fp - yi)**2 - (fm - yi)**2)
        
        # Debug: Check gradient magnitudes
        grad_norm = torch.norm(grads).item()
        
        # Update parameters
        theta -= lr * grads / len(X_train)
        
        # Debug: Check gradient magnitudes
        grad_norm = torch.norm(grads).item()
        
        # Update parameters
        theta -= lr * grads / len(X_train)
        
        # Evaluate on test set (parallelized)
        scores_test = parallel_forward_pass(X_test, theta, model_type, num_workers)
        
        # Compute metrics
        metrics = compute_metrics(scores_test, y_test)
        
        print(
            f"{model_type.upper()} | Epoch {epoch:02d} | "
            f"Acc {metrics['accuracy']:.3f} | "
            f"Prec {metrics['precision']:.3f} | "
            f"Rec {metrics['recall']:.3f} | "
            f"F1 {metrics['f1']:.3f} | "
            f"AUC {metrics['roc_auc']:.3f} | "
            f"GradNorm {grad_norm:.6f}"
        )
    
    return theta


# ============================================================
# Parallel Kernel Model
# ============================================================
def parallel_kernel_predict(X_test, X_train, y_train, num_workers=4):
    """
    Parallel quantum kernel prediction.
    
    Args:
        X_test: Test samples
        X_train: Training samples
        y_train: Training labels
        num_workers: Number of parallel workers
    
    Returns:
        Predictions for test samples
    """
    def predict_single(x):
        return kernel_predict(x, X_train, y_train)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        scores = list(executor.map(predict_single, X_test))
    
    return torch.tensor(scores)


def train_kernel_parallel(num_workers=4):
    """Train quantum kernel model with parallel prediction (SCALED DOWN)."""
    X_train, X_test, y_train, y_test = make_real_dataset()
    
    print(f"\n{'='*60}")
    print(f"Training QUANTUM KERNEL ")
    print(f"Workers: {num_workers}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"{'='*60}\n")
    
    scores = parallel_kernel_predict(X_test, X_train, y_train, num_workers)
    metrics = compute_metrics(scores, y_test)
    
    print("KERNEL TEST METRICS:", metrics)
    return metrics


# ============================================================
# Benchmark Utilities
# ============================================================
def benchmark_parallel_vs_sequential(model_type="deep_vqc", num_workers=4):
    """
    Compare parallel vs sequential training performance.
    
    Args:
        model_type: Model to benchmark
        num_workers: Number of parallel workers
    """
    import time
    from qml_training import train as train_sequential
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_type.upper()}")
    print(f"{'='*60}")
    
    # Sequential training
    print("\n[1/2] Sequential Training...")
    start = time.time()
    train_sequential(model_type)
    seq_time = time.time() - start
    
    # Parallel training
    print(f"\n[2/2] Parallel Training ({num_workers} workers)...")
    start = time.time()
    train_parallel(model_type, num_workers, parallel_mode="gradient")
    par_time = time.time() - start
    
    # Results
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Sequential Time: {seq_time:.2f}s")
    print(f"Parallel Time:   {par_time:.2f}s")
    print(f"Speedup:         {seq_time/par_time:.2f}x")
    print(f"{'='*60}\n")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import sys
    
    # Default: 4 workers
    num_workers = 4
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    
    print(f"\n{'='*60}")
    print(f"PARALLEL QML TRAINING")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")
    
    # Train with gradient parallelization
    print("\n[1/3] Training Deep VQC (Parallel Gradients)...")
    train_parallel("deep_vqc", num_workers=num_workers, parallel_mode="gradient")
    
    print("\n[2/3] Training Noise-Aware QNN (Parallel Gradients)...")
    train_parallel("noise_aware", num_workers=num_workers, parallel_mode="gradient")
    
    print("\n[3/3] Training Quantum Kernel (Parallel Prediction)...")
    train_kernel_parallel(num_workers=num_workers)
    
    # Optional: Uncomment to run benchmark
    print("\n" + "="*60)
    print("Running Benchmark...")
    print("="*60)
    benchmark_parallel_vs_sequential("noise_aware", num_workers=num_workers)
