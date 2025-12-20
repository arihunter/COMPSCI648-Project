#!/usr/bin/env python3
"""
Load and plot per-epoch accuracy history from existing noise sweep data.

Since the existing noise_sweep_moons_ep100_20_0400.csv only has final accuracies,
this script shows you how to regenerate the data with history tracking, or plot
from the new history CSV files when they become available.

Usage:
    python plot_existing_history.py --history-file data/noise_sweep_moons_ep100_20_0400_history.csv
"""

import argparse
import csv
import os
from visualizer import plot_accuracy_vs_epoch


def load_history_from_csv(filepath):
    """
    Load per-epoch history from CSV file.
    
    Expected format: model, encoding, T1, epoch, accuracy
    
    Returns:
        List of result dicts with 'acc_history' populated
    """
    # Group by (model, encoding, T1)
    grouped_data = {}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['model'], row['encoding'], float(row['T1']))
            
            if key not in grouped_data:
                grouped_data[key] = {
                    'model_type': row['model'],
                    'encoding': row['encoding'],
                    'T1': float(row['T1']),
                    'acc_history': []
                }
            
            grouped_data[key]['acc_history'].append(float(row['accuracy']))
    
    return list(grouped_data.values())


def main():
    parser = argparse.ArgumentParser(
        description='Plot accuracy vs epoch from history CSV file'
    )
    parser.add_argument(
        '--history-file',
        type=str,
        help='Path to history CSV file (e.g., data/noise_sweep_moons_ep100_20_0400_history.csv)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='moons',
        help='Dataset name for plot title (default: moons)'
    )
    parser.add_argument(
        '--T1-filter',
        type=float,
        default=None,
        help='Filter by specific T1 value (optional)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save plot to file'
    )
    
    args = parser.parse_args()
    
    if not args.history_file:
        print("Error: --history-file is required")
        print("\nTo generate history data, re-run your experiment:")
        print("  python run_experiments.py --noise-sweep --epochs 100 --T1-values 50 100 200 --plot")
        print("\nThis will create a *_history.csv file with per-epoch accuracy data.")
        return
    
    if not os.path.exists(args.history_file):
        print(f"Error: File not found: {args.history_file}")
        return
    
    print(f"Loading history from: {args.history_file}")
    results = load_history_from_csv(args.history_file)
    print(f"Loaded {len(results)} model configurations")
    
    # Infer epochs from data
    epochs = max(len(r['acc_history']) for r in results) if results else 0
    
    # Create plot
    plot_accuracy_vs_epoch(
        results,
        args.dataset,
        epochs,
        save=args.save,
        T1_filter=args.T1_filter
    )


if __name__ == '__main__':
    main()
