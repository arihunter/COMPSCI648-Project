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
