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
from qml_training import train, EncodingType

# Configuration options
MODEL_TYPES = ["deep_vqc", "noise_aware"]  # kernel doesn't have epoch-based training
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
    print(f"Noise Parameters: T1={T1}, T2={T2}")
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
        choices=['deep_vqc', 'noise_aware'],
        help='Specific models to run (default: all)'
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
    
    args = parser.parse_args()
    
    # Override global configs if specific options provided
    global MODEL_TYPES, ENCODINGS, DATASETS
    
    if args.models:
        MODEL_TYPES = args.models
    if args.encodings:
        ENCODINGS = [EncodingType.ANGLE if e == 'angle' else EncodingType.AMPLITUDE 
                     for e in args.encodings]
    if args.datasets:
        DATASETS = args.datasets
    
    # Run experiments
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
    
    return results


if __name__ == "__main__":
    main()
