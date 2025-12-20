import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def ensure_plots_dir():
    """Create plots directory if it doesn't exist."""
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_training_curves(loss_history, acc_history, dataset="", model_type="", title=None,
                         save=False, T1=None, T2=None, epochs=None):
    """
    Plot training loss and test accuracy curves over epochs.
    
    Args:
        loss_history: List of loss values per epoch
        acc_history: List of accuracy values per epoch
        dataset: Name of the dataset (for title)
        model_type: Name of the model type (for title)
        title: Optional custom title (overrides auto-generated title)
    """
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label="Train Loss", marker='o', markersize=3)
    plt.title(f"{dataset.capitalize()} {model_type.upper()} Train Loss" if not title else f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_history, label="Test Accuracy", marker='o', markersize=3, color='green')
    plt.title(f"{dataset.capitalize()} {model_type.upper()} Test Accuracy" if not title else f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        t1_str = f"_t1{int(T1)}" if T1 is not None else ""
        t2_str = f"_t2{int(T2)}" if T2 is not None else ""
        ep_str = f"_ep{epochs}" if epochs is not None else ""
        filename = f"{model_type}_{dataset}{t1_str}{t2_str}{ep_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {filename}")
    
    plt.show()


def plot_multiple_training_curves(metrics_list, labels=None, title="Training Comparison",
                                   save=False, dataset="", T1=None, T2=None, epochs=None):
    """
    Plot multiple training runs on the same figure for comparison.
    
    Args:
        metrics_list: List of dicts, each containing 'loss_history' and 'acc_history'
        labels: List of labels for each run (optional)
        title: Title for the plot
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(metrics_list))]
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for metrics, label in zip(metrics_list, labels):
        epochs = range(1, len(metrics['loss_history']) + 1)
        plt.plot(epochs, metrics['loss_history'], label=label, marker='o', markersize=2)
    plt.title(f"{title} - Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for metrics, label in zip(metrics_list, labels):
        epochs = range(1, len(metrics['acc_history']) + 1)
        plt.plot(epochs, metrics['acc_history'], label=label, marker='o', markersize=2)
    plt.title(f"{title} - Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        t1_str = f"_t1{int(T1)}" if T1 is not None else ""
        t2_str = f"_t2{int(T2)}" if T2 is not None else ""
        ep_str = f"_ep{epochs}" if epochs is not None else ""
        filename = f"comparison_{dataset}{t1_str}{t2_str}{ep_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {filename}")
    
    plt.show()


def plot_model_comparison(data_dict, dataset_name="", title=None,
                          save=False, T1=None, T2=None, epochs=None):
    """
    Plot accuracy and loss comparison for 4 model configurations.
    
    Args:
        data_dict: Dict mapping model labels to {'loss_history': [...], 'acc_history': [...]}
                   Expected keys: 'VQC Angle', 'VQC Amplitude', 'QNN Angle', 'QNN Amplitude'
        dataset_name: Name of the dataset for title
        title: Optional custom title prefix
        save: If True, save plot to plots/ directory
        T1: T1 noise parameter (for filename)
        T2: T2 noise parameter (for filename)
        epochs: Number of epochs (for filename)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'VQC Angle': '#1f77b4',
        'VQC Amplitude': '#ff7f0e', 
        'QNN Angle': '#2ca02c',
        'QNN Amplitude': '#d62728'
    }
    
    title_prefix = title if title else f"{dataset_name.capitalize()} Dataset"
    
    # Loss plot
    ax1 = axes[0]
    for label, data in data_dict.items():
        epoch_range = range(1, len(data['loss_history']) + 1)
        ax1.plot(epoch_range, data['loss_history'], label=label, 
                 color=colors.get(label, None), linewidth=2, marker='o', markersize=3)
    ax1.set_title(f"{title_prefix} - Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    for label, data in data_dict.items():
        epoch_range = range(1, len(data['acc_history']) + 1)
        ax2.plot(epoch_range, data['acc_history'], label=label,
                 color=colors.get(label, None), linewidth=2, marker='o', markersize=3)
    ax2.set_title(f"{title_prefix} - Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        t1_str = f"_t1{int(T1)}" if T1 is not None else ""
        t2_str = f"_t2{int(T2)}" if T2 is not None else ""
        ep_str = f"_ep{epochs}" if epochs is not None else ""
        filename = f"models_{dataset_name}{t1_str}{t2_str}{ep_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {filename}")
    
    plt.show()


def plot_kernel_comparison(kernel_results, dataset_name, save=False, T1=None, T2=None):
    """
    Plot kernel model performance comparison across encodings.
    
    Args:
        kernel_results: List of kernel result dicts
        dataset_name: Dataset to filter for
        save: If True, save plot to plots/ directory
        T1: T1 noise parameter (for filename)
        T2: T2 noise parameter (for filename)
    """
    # Filter results for this dataset
    filtered = [r for r in kernel_results if r['dataset'] == dataset_name]
    
    if not filtered:
        print(f"No kernel results found for dataset: {dataset_name}")
        return
    
    # Group by encoding
    encodings = []
    accuracies = []
    f1_scores = []
    labels = []
    
    for r in filtered:
        enc = r['encoding']
        noise_label = f"T1={r['T1']}" if r['T1'] != 'noiseless' else "Noiseless"
        labels.append(f"{enc.capitalize()}\n({noise_label})")
        accuracies.append(r['metrics']['accuracy'])
        f1_scores.append(r['metrics']['f1'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#ff7f0e')
    
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name.capitalize()} Dataset - Quantum Kernel Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        t1_str = f"_t1{int(T1)}" if T1 is not None else "_noiseless"
        t2_str = f"_t2{int(T2)}" if T2 is not None else ""
        filename = f"kernel_{dataset_name}{t1_str}{t2_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Kernel plot saved: {filename}")
    
    plt.show()


def plot_kernel_noise_sweep(kernel_results, dataset_name, save=False):
    """
    Plot kernel accuracy vs noise level (T1 values).
    
    Args:
        kernel_results: List of kernel result dicts from noise sweep
        dataset_name: Dataset to filter for
        save: If True, save plot to plots/ directory
    """
    # Filter results for this dataset
    filtered = [r for r in kernel_results if r['dataset'] == dataset_name]
    
    if not filtered:
        print(f"No kernel results found for dataset: {dataset_name}")
        return
    
    # Separate by encoding
    angle_results = [r for r in filtered if r['encoding'] == 'angle']
    amp_results = [r for r in filtered if r['encoding'] == 'amplitude']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, enc_results, enc_name in [(axes[0], angle_results, 'Angle'), 
                                        (axes[1], amp_results, 'Amplitude')]:
        if not enc_results:
            ax.set_title(f'{enc_name} Encoding - No Data')
            continue
        
        # Sort by T1 (put noiseless first)
        t1_values = []
        accuracies = []
        f1_scores = []
        
        for r in sorted(enc_results, key=lambda x: float('inf') if x['T1'] == 'noiseless' else x['T1']):
            t1_label = 'Noiseless' if r['T1'] == 'noiseless' else f"{r['T1']}"
            t1_values.append(t1_label)
            accuracies.append(r['metrics']['accuracy'])
            f1_scores.append(r['metrics']['f1'])
        
        x = np.arange(len(t1_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2ca02c')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#d62728')
        
        ax.set_xlabel('T1 (μs)')
        ax.set_ylabel('Score')
        ax.set_title(f'{dataset_name.capitalize()} - Kernel {enc_name} Encoding')
        ax.set_xticks(x)
        ax.set_xticklabels(t1_values)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        filename = f"kernel_noise_sweep_{dataset_name}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Kernel noise sweep plot saved: {filename}")
    
    plt.show()


def plot_all_models_comparison(training_results, kernel_results, dataset_name, 
                                save=False, T1=None, T2=None, epochs=None):
    """
    Plot comparison of all models (VQC, QNN, Kernel) on same chart.
    
    Args:
        training_results: List of training result dicts (VQC, QNN)
        kernel_results: List of kernel result dicts
        dataset_name: Dataset to filter for
        save: If True, save plot
    """
    # Get final accuracies from training results
    model_scores = {}
    
    for r in training_results:
        if r['dataset'] != dataset_name:
            continue
        model = r['model_type']
        encoding = r['encoding']
        label = f"{'VQC' if model == 'deep_vqc' else 'QNN'} {encoding.capitalize()}"
        model_scores[label] = {
            'accuracy': r['acc_history'][-1],
            'f1': r['final_metrics']['f1']
        }
    
    # Add kernel results
    for r in kernel_results:
        if r['dataset'] != dataset_name:
            continue
        label = f"Kernel {r['encoding'].capitalize()}"
        model_scores[label] = {
            'accuracy': r['metrics']['accuracy'],
            'f1': r['metrics']['f1']
        }
    
    if not model_scores:
        print(f"No results found for dataset: {dataset_name}")
        return
    
    labels = list(model_scores.keys())
    accuracies = [model_scores[l]['accuracy'] for l in labels]
    f1_scores = [model_scores[l]['f1'] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_acc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors_f1 = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color=colors_acc[:len(labels)])
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color=colors_f1[:len(labels)])
    
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name.capitalize()} Dataset - All Models Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        t1_str = f"_t1{int(T1)}" if T1 is not None else ""
        t2_str = f"_t2{int(T2)}" if T2 is not None else ""
        ep_str = f"_ep{epochs}" if epochs is not None else ""
        filename = f"all_models_{dataset_name}{t1_str}{t2_str}{ep_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  All models comparison plot saved: {filename}")
    
    plt.show()


def plot_accuracy_vs_t1(noise_sweep_results, dataset_name, epochs=None, save=False):
    """
    Plot Final Accuracy vs T1 for all models and encodings.
    
    This is the main comparison plot showing how each model/encoding combination
    performs under varying noise levels (T1 relaxation time).
    
    Args:
        noise_sweep_results: List of result dicts from run_unified_noise_sweep
                            Each dict has: model_type, encoding, dataset, T1, T2, accuracy, f1, roc_auc
        dataset_name: Dataset to filter for
        epochs: Number of epochs used (for title/filename)
        save: If True, save plot to plots/ directory
    
    Returns:
        filepath if saved, None otherwise
    """
    # Filter results for this dataset
    filtered = [r for r in noise_sweep_results if r['dataset'] == dataset_name]
    
    if not filtered:
        print(f"No noise sweep results found for dataset: {dataset_name}")
        return None
    
    # Get unique T1 values and sort them
    t1_values = sorted(set(r['T1'] for r in filtered))
    
    # Define model/encoding combinations and their display properties
    configurations = [
        ('deep_vqc', 'angle', 'VQC Angle', '#1f77b4', 'o', '-'),
        ('deep_vqc', 'amplitude', 'VQC Amplitude', '#ff7f0e', 's', '-'),
        ('noise_aware', 'angle', 'QNN Angle', '#2ca02c', '^', '-'),
        ('noise_aware', 'amplitude', 'QNN Amplitude', '#d62728', 'd', '-'),
        ('kernel', 'angle', 'Kernel Angle', '#9467bd', 'v', '--'),
        ('kernel', 'amplitude', 'Kernel Amplitude', '#8c564b', 'p', '--'),
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model_type, encoding, label, color, marker, linestyle in configurations:
        # Get accuracies for this configuration at each T1
        config_results = [r for r in filtered 
                         if r['model_type'] == model_type and r['encoding'] == encoding]
        
        if not config_results:
            continue
        
        # Sort by T1 and extract values
        config_results.sort(key=lambda x: x['T1'])
        x_vals = [r['T1'] for r in config_results]
        y_vals = [r['accuracy'] for r in config_results]
        
        ax.plot(x_vals, y_vals, label=label, color=color, marker=marker, 
                linestyle=linestyle, linewidth=2, markersize=8)
    
    # Formatting
    ax.set_xlabel('T1 Relaxation Time (μs)', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    
    epoch_str = f" ({epochs} epochs)" if epochs else ""
    ax.set_title(f'{dataset_name.capitalize()} Dataset - Accuracy vs Noise Level{epoch_str}', 
                 fontsize=14)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Use log scale for x-axis if range is large
    nonzero_t1 = [t for t in t1_values if t > 0]
    if len(nonzero_t1) > 1 and max(nonzero_t1) / min(nonzero_t1) > 10:
        ax.set_xscale('log')
        ax.set_xlabel('T1 Relaxation Time (μs, log scale)', fontsize=12)
    
    plt.tight_layout()
    
    filepath = None
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        ep_str = f"_ep{epochs}" if epochs else ""
        filename = f"accuracy_vs_t1_{dataset_name}{ep_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Accuracy vs T1 plot saved: {filename}")
    
    plt.show()
    return filepath


def plot_accuracy_vs_t1_dual(noise_sweep_results, epochs=None, save=False):
    """
    Plot Final Accuracy vs T1 for both datasets side by side.
    
    Args:
        noise_sweep_results: List of result dicts from run_unified_noise_sweep
        epochs: Number of epochs used (for title/filename)
        save: If True, save plot to plots/ directory
    
    Returns:
        filepath if saved, None otherwise
    """
    datasets = ['moons', 'real']
    
    # Define model/encoding combinations and their display properties
    configurations = [
        ('deep_vqc', 'angle', 'VQC Angle', '#1f77b4', 'o', '-'),
        ('deep_vqc', 'amplitude', 'VQC Amp', '#ff7f0e', 's', '-'),
        ('noise_aware', 'angle', 'QNN Angle', '#2ca02c', '^', '-'),
        ('noise_aware', 'amplitude', 'QNN Amp', '#d62728', 'd', '-'),
        ('kernel', 'angle', 'Kernel Angle', '#9467bd', 'v', '--'),
        ('kernel', 'amplitude', 'Kernel Amp', '#8c564b', 'p', '--'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, dataset_name in zip(axes, datasets):
        filtered = [r for r in noise_sweep_results if r['dataset'] == dataset_name]
        
        if not filtered:
            ax.set_title(f'{dataset_name.capitalize()} - No Data')
            continue
        
        t1_values = sorted(set(r['T1'] for r in filtered))
        
        for model_type, encoding, label, color, marker, linestyle in configurations:
            config_results = [r for r in filtered 
                             if r['model_type'] == model_type and r['encoding'] == encoding]
            
            if not config_results:
                continue
            
            config_results.sort(key=lambda x: x['T1'])
            x_vals = [r['T1'] for r in config_results]
            y_vals = [r['accuracy'] for r in config_results]
            
            ax.plot(x_vals, y_vals, label=label, color=color, marker=marker, 
                    linestyle=linestyle, linewidth=2, markersize=7)
        
        ax.set_xlabel('T1 (μs)', fontsize=11)
        ax.set_ylabel('Final Accuracy', fontsize=11)
        ax.set_title(f'{dataset_name.capitalize()} Dataset', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        if len(t1_values) > 1 and max(t1_values) / min(t1_values) > 10:
            ax.set_xscale('log')
    
    epoch_str = f" ({epochs} epochs)" if epochs else ""
    fig.suptitle(f'Model Performance vs Noise Level{epoch_str}', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    filepath = None
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        ep_str = f"_ep{epochs}" if epochs else ""
        filename = f"accuracy_vs_t1_comparison{ep_str}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Dual dataset accuracy vs T1 plot saved: {filename}")
    
    plt.show()
    return filepath


def plot_measurement_comparison(normal_counts, kraus_counts, title="Measurement comparison"):
    # All bitstrings that appear in either result
    all_bits = sorted(set(normal_counts.keys()) | set(kraus_counts.keys()))

    # Convert counts to probabilities (handle different shot numbers)
    normal_total = sum(normal_counts.values())
    kraus_total  = sum(kraus_counts.values())

    normal_probs = np.array([normal_counts.get(b, 0) / normal_total for b in all_bits])
    kraus_probs  = np.array([kraus_counts.get(b, 0)  / kraus_total  for b in all_bits])

    x = np.arange(len(all_bits))
    width = 0.4

    plt.figure(figsize=(max(6, len(all_bits) * 0.25), 4))
    plt.bar(x - width/2, normal_probs, width, label="Normal", alpha=0.7)
    plt.bar(x + width/2, kraus_probs,  width, label="Kraus",  alpha=0.7)

    plt.xticks(x, all_bits, rotation=90)
    plt.ylabel("Probability")
    plt.xlabel("Bitstring")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_epoch(results, dataset, epochs, save=False, T1_filter=None):
    """
    Plot accuracy vs epoch for all models (including flat lines for kernel models).
    
    Args:
        results: List of result dicts from run_unified_noise_sweep with 'acc_history'
        dataset: Dataset name for title
        epochs: Number of epochs
        save: If True, save plot to file
        T1_filter: Optional T1 value to filter results (plot only specific T1)
    
    Returns:
        Filepath if save=True, else None
    """
    if not results:
        print(f"No noise sweep results found for dataset: {dataset}")
        return None
    
    # Filter by T1 if specified
    if T1_filter is not None:
        results = [r for r in results if r.get('T1') == T1_filter]
        if not results:
            print(f"No results found for T1={T1_filter}")
            return None
    
    plt.figure(figsize=(12, 7))
    
    # Color and style mapping
    model_colors = {
        'deep_vqc': {'angle': 'blue', 'amplitude': 'dodgerblue'},
        'noise_aware': {'angle': 'red', 'amplitude': 'salmon'},
        'kernel': {'angle': 'green', 'amplitude': 'limegreen'}
    }
    
    model_names = {
        'deep_vqc': 'Deep VQC',
        'noise_aware': 'Noise-Aware QNN',
        'kernel': 'Quantum Kernel'
    }
    
    # Plot each model's accuracy history
    for result in results:
        if 'acc_history' not in result:
            continue
        
        model_type = result['model_type']
        encoding = result['encoding']
        T1 = result.get('T1', 'N/A')
        
        acc_history = result['acc_history']
        epoch_range = range(1, len(acc_history) + 1)
        
        # Create label
        label = f"{model_names.get(model_type, model_type)} ({encoding})"
        if T1_filter is None:
            label += f" T1={T1}"
        
        # Get color
        color = model_colors.get(model_type, {}).get(encoding, 'gray')
        
        # Line style (solid for trainable, dashed for kernel)
        linestyle = '--' if model_type == 'kernel' else '-'
        linewidth = 2 if model_type == 'kernel' else 1.5
        
        plt.plot(epoch_range, acc_history, label=label, color=color, 
                linestyle=linestyle, linewidth=linewidth, marker='o', markersize=3, alpha=0.8)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    
    title = f"Model Accuracy vs Epoch - {dataset.capitalize()} Dataset"
    if T1_filter is not None:
        title += f" (T1={T1_filter} µs)"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Improved legend: place outside plot, organized by model type
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    
    # Sort labels by model type for better organization
    model_order = {'Deep VQC': 0, 'Noise-Aware QNN': 1, 'Quantum Kernel': 2}
    def sort_key(item):
        label = item[1]
        for model_name, order in model_order.items():
            if model_name in label:
                return order
        return 999
    
    sorted_items = sorted(zip(handles, labels), key=sort_key)
    handles, labels = zip(*sorted_items) if sorted_items else ([], [])
    
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=10, framealpha=0.95, edgecolor='black', title='Models',
              title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir()
        timestamp = datetime.now().strftime("%d_%H%M")
        t1_suffix = f"_t1{T1_filter}" if T1_filter else "_all_t1"
        filename = f"accuracy_vs_epoch_{dataset}_ep{epochs}{t1_suffix}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {filepath}")
        return filepath
    else:
        plt.show()
        return None


# # Example usage with your sample (pretend it's from one of the simulators)
# normal_counts = {'00011': 33, '11010': 57, '11101': 73, '01101': 19, '11111': 108,
#                  '10010': 84, '10100': 22, '10111': 56, '11001': 33, '01000': 34,
#                  '00101': 48, '01110': 67, '11110': 123, '11100': 17, '01011': 23,
#                  '10101': 19, '01010': 54, '01100': 14, '00110': 11, '00000': 23,
#                  '01111': 10, '01001': 15, '11011': 14, '10110': 21, '00001': 5,
#                  '11000': 13, '10011': 14, '00100': 10, '00010': 3, '10000': 1}

# # For demonstration, just reuse the same dict as "kraus_counts"
# kraus_counts = normal_counts.copy()

# plot_measurement_comparison(normal_counts, kraus_counts,
#                             title="Normal vs Kraus measurement distributions")
