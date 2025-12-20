# COMPSCI 648 - Quantum Machine Learning Project

Quantum machine learning for binary classification using variational quantum circuits, built with PyTorch.

## Project Structure

```
quantum_simulator.py      # States, gates, measurements
qml_training.py           # VQC training pipelines
run_experiments.py        # CLI experiment runner
error_kraus.py            # T1/T2 noise via Kraus operators
metrics.py                # Accuracy, F1, ROC-AUC
visualizer.py             # Training curve plots
demos/circuit_visualization.ipynb
data/                     # Output CSVs
tests/
```

## Encoding

- **Angle**: Feature θ → RY(θ) rotation. N features use N qubits.
- **Amplitude**: Features as state amplitudes. 2^N values use N qubits.

## Noise Model

T1/T2 thermal relaxation applied after each gate:
- T1: amplitude damping (|1⟩ → |0⟩ decay)
- T2: phase damping (coherence loss)
 
 Circuit error-adding functionality:
 - Time-based idle noise between gates via `gate_durations` (μs)
 - Optional decoherence during gates using `gate_noise_fraction` (0–1, default to 1)
 - Kraus-based simulation with `T_φ` derived from `T1,T2`; enforces `T2 ≤ 2·T1`
 - Units: all times in microseconds; typical: H≈0.025 μs, CNOT≈0.2 μs
 - Relaxation model based on superconducting transmon qubit

## Models

**Deep VQC**: Multi-layer variational circuit with RY, RZ, CNOT layers.

**Noise-Aware VQC**: Same architecture, trained with simulated T1/T2 noise using density matrix evolution.

**Quantum Kernel**: Feature-map based classifier using quantum state overlap. No gradient-based training—predicts labels via kernel sum against training set. Useful for comparing encoding methods under varying noise.

## Usage

### Trainable Models (VQC, Noise-Aware QNN)

```bash
# Default run (all models, all encodings, all datasets)
python run_experiments.py

# Custom parameters
python run_experiments.py --epochs 50 --T1 50 --T2 100

# Specific configuration
python run_experiments.py --models deep_vqc --encodings angle --datasets moons --plot
```

### Quantum Kernel Model

```bash
# Run kernel with default noise parameters
python run_experiments.py --kernel

# Run kernel without noise (noiseless simulation)
python run_experiments.py --kernel-noiseless

# Run kernel with noise sweep (tests T1 = 25, 50, 100, 200, 500 μs)
python run_experiments.py --kernel-noise-sweep --plot

# Run kernel for specific encoding/dataset
python run_experiments.py --kernel --encodings angle --datasets moons

# Compare kernel results with plots
python run_experiments.py --kernel --kernel-compare
```

### Combined Runs

```bash
# Run all models including kernel
python run_experiments.py --models deep_vqc noise_aware kernel --plot

# Run specific models with kernel
python run_experiments.py --models kernel noise_aware --encodings angle --datasets moons
```

### Unified Noise Sweep (Accuracy vs T1 Plot)

Run all models (VQC, QNN, Kernel) across varying T1 noise levels and generate comparison plots:

```bash
# Run noise sweep with default T1 values (25, 50, 100, 200, 500 μs)
python run_experiments.py --noise-sweep --plot

# Custom T1 values and epochs
python run_experiments.py --noise-sweep --epochs 10 --T1-values 50 100 200 --plot

# Noise sweep on specific dataset
python run_experiments.py --noise-sweep --datasets moons --epochs 25 --plot

# Custom T2/T1 ratio (default is 2.0)
python run_experiments.py --noise-sweep --T2-ratio 1.5 --plot
```

This generates a plot with **T1 (μs) on x-axis** and **Final Accuracy on y-axis** for all 6 configurations:
- VQC Angle, VQC Amplitude
- QNN Angle, QNN Amplitude  
- Kernel Angle, Kernel Amplitude

### Accuracy vs Epoch (Per-Model Lines)

When running noise sweeps with `--plot`, the runner also saves per-epoch accuracy histories and generates accuracy-vs-epoch plots (kernel appears as a flat dashed line):

```bash
# Generate sweep + per-epoch history and plots
python run_experiments.py --noise-sweep --epochs 100 --T1-values 50 100 200 --plot

# Plot from a saved history CSV later (optional)
python plot_existing_history.py --history-file data/noise_sweep_moons_ep100_<timestamp>_history.csv --dataset moons --save
```

Artifacts:
- Data: `data/noise_sweep_<dataset>_ep<epochs>_<timestamp>_history.csv` (columns: model, encoding, T1, epoch, accuracy)
- Plots: saved under `plots/` with filenames like `accuracy_vs_epoch_<dataset>_ep<epochs>_t1<T1>_<timestamp>.png`

## CLI Arguments

| Arg | Default | Options | Description |
|-----|---------|---------|-------------|
| `--epochs` | 25 | int | Training epochs for VQC/QNN |
| `--T1` | 100 | float | T1 relaxation (μs) |
| `--T2` | 200 | float | T2 dephasing (μs) |
| `--models` | all | `deep_vqc`, `noise_aware`, `kernel` | Models to run |
| `--encodings` | all | `angle`, `amplitude` | Encoding methods |
| `--datasets` | all | `real`, `moons` | Datasets to use |
| `--plot` | off | flag | Save comparison plots |
| `--kernel` | off | flag | Run kernel experiments |
| `--kernel-noiseless` | off | flag | Run kernel without noise |
| `--kernel-noise-sweep` | off | flag | Run kernel with varying noise levels |
| `--kernel-compare` | off | flag | Plot kernel comparison charts |
| `--noise-sweep` | off | flag | Run all models with varying T1 values |
| `--T1-values` | 25 50 100 200 500 | floats | T1 values (μs) for noise sweep |
| `--T2-ratio` | 2.0 | float | T2/T1 ratio for noise sweep |

## Datasets

- **real**: Breast cancer (UCI), PCA to 2-4 features
- **moons**: Synthetic two-moons, 2D

## Demo Notebook

`demos/circuit_visualization.ipynb` shows encoding comparisons, kernel matrices, and noise effects on fidelity.

## Install

```bash
pip install -e .
```

Requires Python 3.11, PyTorch ≥2.0, NumPy, Matplotlib, scikit-learn.

## Environment Setup

Use `uv` to create a local 3.11 environment (reads `pyproject.toml`):

```bash
uv sync
source .venv/bin/activate
```

Or run without activating:

```bash
uv run python run_experiments.py --help
```
