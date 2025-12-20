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

## Usage

```bash
# Default run
python run_experiments.py

# Custom parameters
python run_experiments.py --epochs 50 --T1 50 --T2 100

# Specific configuration
python run_experiments.py --models deep_vqc --encodings angle --datasets moons --plot
```

| Arg | Default | Options |
|-----|---------|---------|
| `--epochs` | 25 | |
| `--T1` | 100 | T1 relaxation (μs) |
| `--T2` | 200 | T2 dephasing (μs) |
| `--models` | all | `deep_vqc`, `noise_aware` |
| `--encodings` | all | `angle`, `amplitude` |
| `--datasets` | all | `real`, `moons` |
| `--plot` | off | Save comparison plots |

## Datasets

- **real**: Breast cancer (UCI), PCA to 2-4 features
- **moons**: Synthetic two-moons, 2D

## Demo Notebook

`demos/circuit_visualization.ipynb` shows encoding comparisons, kernel matrices, and noise effects on fidelity.

## Install

```bash
pip install -e .
```

Requires Python ≥3.8, PyTorch ≥2.0, NumPy, Matplotlib, scikit-learn.
