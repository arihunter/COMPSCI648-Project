import matplotlib.pyplot as plt
import numpy as np

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
