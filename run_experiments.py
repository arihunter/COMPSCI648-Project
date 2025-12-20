"""
Experiment Runner for QML Training

Runs all training configurations with specified parameters and saves
results to CSV files in the data/ folder.

Usage:
    python run_experiments.py --epochs 25 --T1 100 --T2 200
    python run_experiments.py --epochs 50 --T1 50 --T2 100 --output_prefix "noisy_"
"""

import os
import csv
import argparse
from datetime import datetime
from itertools import product

from constants import DEFAULT_T1, DEFAULT_T2
from qml_training import (
    train, EncodingType, kernel_predict, kernel_predict_batch, compute_metrics,
    make_real_dataset, make_moons_dataset
)

# Configuration options
MODEL_TYPES = ["deep_vqc", "noise_aware"]  # kernel handled separately (no epoch-based training)
KERNEL_MODEL = "kernel"  # Kernel model runs once per configuration
ENCODINGS = [EncodingType.ANGLE, EncodingType.AMPLITUDE]
DATASETS = ["real", "moons"]


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_csv_history(filepath):
    """
    Load training history from CSV file.
    
    Returns:
        dict with 'loss_history' and 'acc_history' lists
    """
    loss_history = []
    acc_history = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss_history.append(float(row['loss']))
            acc_history.append(float(row['acc']))
    return {'loss_history': loss_history, 'acc_history': acc_history}


def find_matching_csvs(data_dir, dataset, T1, T2, epochs):
    """
    Find CSV files matching the given parameters.
    
    Returns:
        dict mapping model labels to file paths
    """
    pattern_map = {
        'VQC Angle': f"deep_vqc_{dataset}_t1{int(T1)}_t2{int(T2)}_ep{epochs}",
        'VQC Amplitude': f"deep_vqc_{dataset}_t1{int(T1)}_t2{int(T2)}_ep{epochs}",
        'QNN Angle': f"noise_aware_{dataset}_t1{int(T1)}_t2{int(T2)}_ep{epochs}",
        'QNN Amplitude': f"noise_aware_{dataset}_t1{int(T1)}_t2{int(T2)}_ep{epochs}",
    }
    
    # Actually we need to differentiate by encoding in the filename
    # Current format: {model}_{dataset}_t1{T1}_t2{T2}_ep{epochs}_{date}.csv
    # We need to find files and match by model type
    
    found_files = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv') or filename.startswith('summary'):
            continue
        
        # Parse filename: model_dataset_t1X_t2Y_epZ_date.csv
        parts = filename.replace('.csv', '').split('_')
        if len(parts) < 6:
            continue
            
        file_model = parts[0]
        if parts[0] == 'deep':
            file_model = 'deep_vqc'
            parts = ['deep_vqc'] + parts[2:]  # rejoin
        elif parts[0] == 'noise':
            file_model = 'noise_aware'
            parts = ['noise_aware'] + parts[2:]
        
        file_dataset = parts[1]
        
        # Check if matches our criteria
        if file_dataset != dataset:
            continue
        if f"t1{int(T1)}" not in filename or f"t2{int(T2)}" not in filename:
            continue
        if f"ep{epochs}" not in filename:
            continue
        
        # Determine encoding from the file content or use latest file
        filepath = os.path.join(data_dir, filename)
        
        if file_model == 'deep_vqc':
            # We'll need to track both VQC files and pick based on order
            if 'VQC' not in str(found_files.keys()):
                found_files[f"VQC_{filename}"] = filepath
        elif file_model == 'noise_aware':
            if 'QNN' not in str(found_files.keys()):
                found_files[f"QNN_{filename}"] = filepath
    
    return found_files


def plot_from_results(results, dataset, T1=DEFAULT_T1, T2=DEFAULT_T2, epochs=25, save=False):
    """
    Plot comparison from in-memory results.
    
    Args:
        results: List of result dicts from run_all_experiments
        dataset: Dataset to filter for
        T1: T1 noise parameter
        T2: T2 noise parameter
        epochs: Number of epochs
        save: If True, save plots to plots/ directory
    """
    from visualizer import plot_model_comparison
    
    data_dict = {}
    for r in results:
        if r['dataset'] != dataset:
            continue
        
        model = r['model_type']
        encoding = r['encoding']
        
        if model == 'deep_vqc':
            label = f"VQC {encoding.capitalize()}"
        else:
            label = f"QNN {encoding.capitalize()}"
        
        data_dict[label] = {
            'loss_history': r['loss_history'],
            'acc_history': r['acc_history']
        }
    
    if data_dict:
        plot_model_comparison(data_dict, dataset_name=dataset, 
                              save=save, T1=T1, T2=T2, epochs=epochs)


def save_training_history_csv(metrics, filepath):
    """
    Save epoch-by-epoch training history to CSV.
    
    Columns: epoch, loss, acc
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'acc'])
        for epoch, (loss, acc) in enumerate(zip(metrics['loss_history'], metrics['acc_history']), start=1):
            writer.writerow([epoch, f'{loss:.6f}', f'{acc:.4f}'])


def save_summary_csv(all_results, filepath):
    """
    Save summary of all experiments to a single CSV.
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'enc', 'data', 'loss', 'acc', 'f1', 'auc'])
        for r in all_results:
            writer.writerow([
                r['model_type'],
                r['encoding'],
                r['dataset'],
                f"{r['loss_history'][-1]:.4f}",
                f"{r['acc_history'][-1]:.4f}",
                f"{r['final_metrics']['f1']:.4f}",
                f"{r['final_metrics']['roc_auc']:.4f}"
            ])


def save_kernel_results_csv(results, filepath):
    """
    Save kernel model results to CSV.
    
    Columns: encoding, dataset, T1, T2, accuracy, precision, recall, f1, roc_auc
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['encoding', 'dataset', 'T1', 'T2', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        for r in results:
            writer.writerow([
                r['encoding'],
                r['dataset'],
                r['T1'],
                r['T2'],
                f"{r['metrics']['accuracy']:.4f}",
                f"{r['metrics']['precision']:.4f}",
                f"{r['metrics']['recall']:.4f}",
                f"{r['metrics']['f1']:.4f}",
                f"{r['metrics']['roc_auc']:.4f}"
            ])


import torch

def run_kernel_experiment(encoding=EncodingType.ANGLE, dataset="moons", T1=None, T2=None):
    """
    Run quantum kernel model and return metrics.
    
    Args:
        encoding: EncodingType.ANGLE or EncodingType.AMPLITUDE
        dataset: 'real' or 'moons'
        T1: T1 relaxation time (None for noiseless)
        T2: T2 dephasing time (None for noiseless)
    
    Returns:
        dict with metrics and configuration
    """
    # Load dataset
    if dataset == "real":
        X_train, X_test, y_train, y_test = make_real_dataset(encoding=encoding)
    elif dataset == "moons":
        X_train, X_test, y_train, y_test = make_moons_dataset(encoding=encoding)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Run batch kernel predictions with cached feature maps
    scores = kernel_predict_batch(X_test, X_train, y_train, encoding, T1, T2)
    
    # Compute metrics
    metrics = compute_metrics(scores, y_test)
    
    return {
        'model_type': 'kernel',
        'encoding': encoding.value,
        'dataset': dataset,
        'T1': T1 if T1 is not None else 'noiseless',
        'T2': T2 if T2 is not None else 'noiseless',
        'metrics': metrics
    }


def run_kernel_noise_sweep(encoding=EncodingType.ANGLE, dataset="moons", 
                            T1_values=None, T2_ratio=2.0):
    """
    Run kernel model under varying noise levels.
    
    Args:
        encoding: EncodingType.ANGLE or EncodingType.AMPLITUDE
        dataset: 'real' or 'moons'
        T1_values: List of T1 values to test (default: [25, 50, 100, 200, 500])
        T2_ratio: Ratio of T2 to T1 (default: 2.0, so T2 = 2*T1)
    
    Returns:
        List of result dicts for each noise level
    """
    if T1_values is None:
        T1_values = [25, 50, 100, 200, 500]
    
    results = []
    
    # First run noiseless
    print(f"  Running kernel (noiseless)...")
    result = run_kernel_experiment(encoding, dataset, T1=None, T2=None)
    results.append(result)
    print(f"    Accuracy: {result['metrics']['accuracy']:.4f}")
    
    # Then run with varying noise levels
    for T1 in T1_values:
        T2 = T1 * T2_ratio
        print(f"  Running kernel (T1={T1}, T2={T2})...")
        result = run_kernel_experiment(encoding, dataset, T1=T1, T2=T2)
        results.append(result)
        print(f"    Accuracy: {result['metrics']['accuracy']:.4f}")
    
    return results


def run_all_kernel_experiments(T1=None, T2=None, noise_sweep=False, T1_values=None):
    """
    Run kernel model for all encoding/dataset combinations.
    
    Args:
        T1: T1 relaxation time (None for noiseless)
        T2: T2 dephasing time (None for noiseless)
        noise_sweep: If True, run noise sweep instead of single T1/T2
        T1_values: List of T1 values for noise sweep
    
    Returns:
        List of all result dictionaries
    """
    data_dir = ensure_data_dir()
    all_results = []
    timestamp = datetime.now().strftime("%d_%H%M")
    
    configurations = list(product(ENCODINGS, DATASETS))
    total_configs = len(configurations)
    
    print(f"\n{'='*60}")
    print(f"Quantum Kernel Model Experiments")
    print(f"{'='*60}")
    if noise_sweep:
        print(f"Mode: Noise sweep")
        print(f"T1 values: {T1_values if T1_values else [25, 50, 100, 200, 500]} µs")
    else:
        print(f"Noise Parameters: T1={T1} µs, T2={T2} µs")
    print(f"Total configurations: {total_configs}")
    print(f"Output directory: {data_dir}")
    print(f"{'='*60}\n")
    
    for idx, (encoding, dataset) in enumerate(configurations, start=1):
        config_name = f"kernel_{encoding.value}_{dataset}"
        print(f"\n[{idx}/{total_configs}] Running: {config_name}")
        print("-" * 40)
        
        try:
            if noise_sweep:
                results = run_kernel_noise_sweep(encoding, dataset, T1_values)
                all_results.extend(results)
            else:
                result = run_kernel_experiment(encoding, dataset, T1, T2)
                all_results.append(result)
                print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
                print(f"  F1 Score: {result['metrics']['f1']:.4f}")
        except Exception as e:
            print(f"  Error running {config_name}: {e}")
            continue
    
    # Save results
    if all_results:
        if noise_sweep:
            filename = f"kernel_noise_sweep_{timestamp}.csv"
        else:
            t1_str = f"t1{int(T1)}" if T1 is not None else "noiseless"
            t2_str = f"_t2{int(T2)}" if T2 is not None else ""
            filename = f"kernel_{t1_str}{t2_str}_{timestamp}.csv"
        
        filepath = os.path.join(data_dir, filename)
        save_kernel_results_csv(all_results, filepath)
        print(f"\n{'='*60}")
        print(f"Results saved: {filename}")
        print(f"Total successful runs: {len(all_results)}")
        print(f"{'='*60}\n")
    
    return all_results


def save_noise_sweep_csv(results, filepath):
    """
    Save unified noise sweep results (all models) to CSV.
    
    Columns: model, encoding, dataset, T1, T2, accuracy, f1, roc_auc
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'encoding', 'dataset', 'T1', 'T2', 'accuracy', 'f1', 'roc_auc'])
        for r in results:
            writer.writerow([
                r['model_type'],
                r['encoding'],
                r['dataset'],
                r['T1'],
                r['T2'],
                f"{r['accuracy']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['roc_auc']:.4f}"
            ])


def run_unified_noise_sweep(epochs=25, dataset="moons", T1_values=None, T2_ratio=2.0):
    """
    Run ALL models (VQC, QNN, Kernel) across varying T1 noise levels.
    
    This produces the data needed for the "Final Accuracy vs T1" comparison plot.
    
    Args:
        epochs: Number of training epochs for VQC/QNN
        dataset: Dataset to use ('real' or 'moons')
        T1_values: List of T1 values in µs (default: [25, 50, 100, 200, 500])
        T2_ratio: Ratio of T2 to T1 (default: 2.0)
    
    Returns:
        List of result dicts with unified format
    """
    if T1_values is None:
        T1_values = [25, 50, 100, 200, 500]
    
    data_dir = ensure_data_dir()
    all_results = []
    timestamp = datetime.now().strftime("%d_%H%M")
    
    all_models = ["deep_vqc", "noise_aware", "kernel"]
    all_encodings = [EncodingType.ANGLE, EncodingType.AMPLITUDE]
    
    total_runs = len(all_models) * len(all_encodings) * len(T1_values)
    
    print(f"\n{'='*60}")
    print(f"Unified Noise Sweep Experiment")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs} (for trainable models)")
    print(f"T1 values: {T1_values} µs")
    print(f"T2 ratio: {T2_ratio} (T2 = T1 × {T2_ratio})")
    print(f"Models: {all_models}")
    print(f"Encodings: {[e.value for e in all_encodings]}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*60}\n")
    
    run_count = 0
    for encoding in all_encodings:
        for model_type in all_models:
            for T1 in T1_values:
                T2 = T1 * T2_ratio
                run_count += 1
                
                config_name = f"{model_type}_{encoding.value}_T1={T1}"
                print(f"\n[{run_count}/{total_runs}] {config_name}")
                print("-" * 40)
                
                try:
                    if model_type == "kernel":
                        # Kernel: batch prediction with cached feature maps (fast!)
                        result = run_kernel_experiment(encoding, dataset, T1, T2)
                        unified_result = {
                            'model_type': 'kernel',
                            'encoding': encoding.value,
                            'dataset': dataset,
                            'T1': T1,
                            'T2': T2,
                            'accuracy': result['metrics']['accuracy'],
                            'f1': result['metrics']['f1'],
                            'roc_auc': result['metrics']['roc_auc'],
                            'epochs': 0
                        }
                    else:
                        # Train VQC or QNN
                        metrics = train(
                            model_type=model_type,
                            encoding=encoding,
                            epochs=epochs,
                            dataset=dataset,
                            record_metrics=True,
                            T1=T1,
                            T2=T2
                        )
                        
                        if metrics is None:
                            print(f"  Warning: No metrics returned")
                            continue
                        
                        unified_result = {
                            'model_type': model_type,
                            'encoding': encoding.value,
                            'dataset': dataset,
                            'T1': T1,
                            'T2': T2,
                            'accuracy': metrics['acc_history'][-1],
                            'f1': metrics['final_metrics']['f1'],
                            'roc_auc': metrics['final_metrics']['roc_auc'],
                            'epochs': epochs
                        }
                    
                    all_results.append(unified_result)
                    print(f"  T1={T1} µs, T2={T2} µs → Accuracy: {unified_result['accuracy']:.4f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
    
    # Save results
    if all_results:
        filename = f"noise_sweep_{dataset}_ep{epochs}_{timestamp}.csv"
        filepath = os.path.join(data_dir, filename)
        save_noise_sweep_csv(all_results, filepath)
        print(f"\n{'='*60}")
        print(f"Noise sweep results saved: {filename}")
        print(f"Total successful runs: {len(all_results)}/{total_runs}")
        print(f"{'='*60}\n")
    
    return all_results


def run_all_experiments(epochs=25, T1=DEFAULT_T1, T2=DEFAULT_T2):
    """
    Run all training configurations and save results.
    
    Args:
        epochs: Number of training epochs
        T1: T1 relaxation time for noise model (µs)
        T2: T2 dephasing time for noise model (µs)
    
    Returns:
        List of all results dictionaries
    """
    data_dir = ensure_data_dir()
    all_results = []
    
    # Generate timestamp for this run (day_hour_min)
    timestamp = datetime.now().strftime("%d_%H%M")
    
    # All configurations to run
    configurations = list(product(MODEL_TYPES, ENCODINGS, DATASETS))
    total_configs = len(configurations)
    
    print(f"\n{'='*60}")
    print(f"QML Experiment Runner")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Noise Parameters: T1={T1} µs, T2={T2} µs")
    print(f"Total configurations: {total_configs}")
    print(f"Output directory: {data_dir}")
    print(f"{'='*60}\n")
    
    for idx, (model_type, encoding, dataset) in enumerate(configurations, start=1):
        config_name = f"{model_type}_{encoding.value}_{dataset}"
        print(f"\n[{idx}/{total_configs}] Running: {config_name}")
        print("-" * 40)
        
        try:
            # Run training with record_metrics=True
            metrics = train(
                model_type=model_type,
                encoding=encoding,
                epochs=epochs,
                dataset=dataset,
                record_metrics=True,
                T1=T1,
                T2=T2
            )
            
            if metrics is None:
                print(f"  Warning: No metrics returned for {config_name}")
                continue
            
            # Add noise parameters to metrics
            metrics['T1'] = T1
            metrics['T2'] = T2
            
            all_results.append(metrics)
            
            # Save individual training history CSV
            # Format: {model}_{dataset}_t1{T1}_t2{T2}_ep{epochs}_{date}.csv
            history_filename = f"{model_type}_{dataset}_t1{int(T1)}_t2{int(T2)}_ep{epochs}_{timestamp}.csv"
            history_filepath = os.path.join(data_dir, history_filename)
            save_training_history_csv(metrics, history_filepath)
            print(f"  Saved: {history_filename}")
            
        except Exception as e:
            print(f"  Error running {config_name}: {e}")
            continue
    
    # Save summary CSV with all results
    if all_results:
        # Format: summary_t1{T1}_t2{T2}_ep{epochs}_{date}.csv
        summary_filename = f"summary_t1{int(T1)}_t2{int(T2)}_ep{epochs}_{timestamp}.csv"
        summary_filepath = os.path.join(data_dir, summary_filename)
        save_summary_csv(all_results, summary_filepath)
        print(f"\n{'='*60}")
        print(f"Summary saved: {summary_filename}")
        print(f"Total successful runs: {len(all_results)}/{total_configs}")
        print(f"{'='*60}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run QML training experiments and save results to CSV"
    )
    parser.add_argument(
        '--epochs', type=int, default=25,
        help='Number of training epochs (default: 25)'
    )
    parser.add_argument(
        '--T1', type=float, default=DEFAULT_T1,
        help=f'T1 relaxation time in µs for noise model (default: {DEFAULT_T1})'
    )
    parser.add_argument(
        '--T2', type=float, default=DEFAULT_T2,
        help=f'T2 dephasing time in µs for noise model (default: {DEFAULT_T2})'
    )
    parser.add_argument(
        '--models', type=str, nargs='+', default=None,
        choices=['deep_vqc', 'noise_aware', 'kernel'],
        help='Specific models to run (default: all trainable models)'
    )
    parser.add_argument(
        '--encodings', type=str, nargs='+', default=None,
        choices=['angle', 'amplitude'],
        help='Specific encodings to run (default: all)'
    )
    parser.add_argument(
        '--datasets', type=str, nargs='+', default=None,
        choices=['real', 'moons'],
        help='Specific datasets to run (default: all)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Plot accuracy and loss comparison after training'
    )
    parser.add_argument(
        '--kernel', action='store_true',
        help='Run quantum kernel model experiments'
    )
    parser.add_argument(
        '--kernel-noise-sweep', action='store_true',
        help='Run kernel model with varying noise levels'
    )
    parser.add_argument(
        '--kernel-noiseless', action='store_true',
        help='Run kernel model without noise'
    )
    parser.add_argument(
        '--kernel-compare', action='store_true',
        help='Plot kernel results comparison with other models'
    )
    parser.add_argument(
        '--noise-sweep', action='store_true',
        help='Run unified noise sweep across all models (VQC, QNN, Kernel) with varying T1'
    )
    parser.add_argument(
        '--T1-values', type=float, nargs='+', default=None,
        help='T1 values in µs for noise sweep (default: 25 50 100 200 500)'
    )
    parser.add_argument(
        '--T2-ratio', type=float, default=2.0,
        help='T2/T1 ratio for noise sweep (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Override global configs if specific options provided
    global MODEL_TYPES, ENCODINGS, DATASETS
    
    if args.encodings:
        ENCODINGS = [EncodingType.ANGLE if e == 'angle' else EncodingType.AMPLITUDE 
                     for e in args.encodings]
    if args.datasets:
        DATASETS = args.datasets
    
    results = []
    kernel_results = []
    noise_sweep_results = []
    
    # Handle unified noise sweep (highest priority - runs all models)
    if args.noise_sweep:
        datasets_to_run = args.datasets if args.datasets else ['moons', 'real']
        for dataset in datasets_to_run:
            sweep_results = run_unified_noise_sweep(
                epochs=args.epochs,
                dataset=dataset,
                T1_values=args.T1_values,
                T2_ratio=args.T2_ratio
            )
            noise_sweep_results.extend(sweep_results)
        
        # Plot noise sweep results
        if args.plot and noise_sweep_results:
            from visualizer import plot_accuracy_vs_t1
            for dataset in datasets_to_run:
                dataset_results = [r for r in noise_sweep_results if r['dataset'] == dataset]
                if dataset_results:
                    plot_accuracy_vs_t1(dataset_results, dataset, 
                                       epochs=args.epochs, save=True)
        
        return results, kernel_results, noise_sweep_results
    
    # Handle kernel-only runs
    run_kernel_only = args.kernel or args.kernel_noise_sweep or args.kernel_noiseless
    run_kernel_model = 'kernel' in (args.models or [])
    
    if run_kernel_model and args.models:
        # Remove kernel from models list (handled separately)
        args.models = [m for m in args.models if m != 'kernel']
    
    if args.models:
        MODEL_TYPES = args.models
    
    # Run kernel experiments if requested
    if run_kernel_only or run_kernel_model:
        if args.kernel_noise_sweep:
            kernel_results = run_all_kernel_experiments(noise_sweep=True, T1_values=args.T1_values)
        elif args.kernel_noiseless:
            kernel_results = run_all_kernel_experiments(T1=None, T2=None)
        else:
            kernel_results = run_all_kernel_experiments(T1=args.T1, T2=args.T2)
        
        # Plot kernel comparison if requested
        if args.kernel_compare or args.plot:
            from visualizer import plot_kernel_comparison, plot_kernel_noise_sweep
            datasets_to_plot = args.datasets if args.datasets else ['real', 'moons']
            for dataset in datasets_to_plot:
                if args.kernel_noise_sweep:
                    plot_kernel_noise_sweep(kernel_results, dataset, save=True)
                else:
                    plot_kernel_comparison(kernel_results, dataset, save=True,
                                          T1=args.T1, T2=args.T2)
    
    # Run trainable model experiments (unless only kernel was requested)
    if not run_kernel_only or (args.models and len(args.models) > 0):
        if not run_kernel_only:  # Only run if not kernel-only mode
            results = run_all_experiments(
                epochs=args.epochs,
                T1=args.T1,
                T2=args.T2
            )
            
            # Plot if requested (always save when --plot is used)
            if args.plot and results:
                datasets_to_plot = args.datasets if args.datasets else ['real', 'moons']
                for dataset in datasets_to_plot:
                    plot_from_results(results, dataset, 
                                      T1=args.T1, T2=args.T2, epochs=args.epochs, save=True)
    
    return results, kernel_results, noise_sweep_results


if __name__ == "__main__":
    main()
