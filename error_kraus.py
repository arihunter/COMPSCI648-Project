from typing import List, Tuple, Dict
import math
import torch

# Identity
I2 = torch.eye(2, dtype=torch.cfloat)

# ============================================================
# Helper: clamp probabilities to [0, 1]
# ============================================================
def clamp_prob(p: float) -> float:
    return max(0.0, min(1.0, p))

# ============================================================
# Thermal relaxation error rates (T1 / T2)
# ============================================================
def thermal_relaxation_error_rate(
    T1: float,
    T2: float,
    idle_time: float
) -> tuple[float, float]:
    """
    Given T1, T2, and idle time Δt, return (λ1, λ2):

    λ1 : amplitude damping probability
    λ2 : pure dephasing probability

    Physics:
      λ1 = 1 - exp(-Δt / T1)
      exp(-Δt/T2) = exp(-Δt/(2T1)) * (1 - 2λ2)
    """
    if idle_time <= 0.0:
        return 0.0, 0.0

    # --- Amplitude damping (T1) ---
    if T1 <= 0.0 or math.isinf(T1):
        λ1 = 0.0
    else:
        λ1 = 1.0 - math.exp(-idle_time / T1)

    λ1 = clamp_prob(λ1)

    # --- Pure dephasing (T2) ---
    if T2 <= 0.0 or math.isinf(T2):
        λ2 = 0.0
    else:
        if T1 <= 0.0 or math.isinf(T1):
            λ2 = (1.0 - math.exp(-idle_time / T2)) / 2.0
        else:
            exp_arg = -idle_time / T2 + idle_time / (2.0 * T1)
            λ2 = (1.0 - math.exp(exp_arg)) / 2.0

    λ2 = clamp_prob(λ2)

    return λ1, λ2

# ============================================================
# Kraus operators: Amplitude damping
# ============================================================
def amplitude_damping_kraus(λ1: float) -> List[torch.Tensor]:
    """
    Single-qubit amplitude damping channel.
    |1> -> |0> with probability λ1
    """
    λ1 = clamp_prob(λ1)

    if λ1 == 0.0:
        return [torch.eye(2, dtype=torch.cfloat)]

    p = λ1

    K0 = torch.tensor(
        [[1.0, 0.0],
         [0.0, math.sqrt(1.0 - p)]],
        dtype=torch.cfloat
    )

    K1 = torch.tensor(
        [[0.0, math.sqrt(p)],
         [0.0, 0.0]],
        dtype=torch.cfloat
    )

    return [K0, K1]

# ============================================================
# Kraus operators: Phase damping
# ============================================================
def phase_damping_kraus(λ2: float) -> List[torch.Tensor]:
    """
    Single-qubit pure dephasing channel.
    Off-diagonals decay, populations unchanged.
    """
    λ2 = clamp_prob(λ2)

    if λ2 == 0.0:
        return [torch.eye(2, dtype=torch.cfloat)]

    p = λ2
    Z = torch.tensor([[1.0, 0.0],
                      [0.0, -1.0]], dtype=torch.cfloat)

    K0 = math.sqrt(1.0 - p) * I2
    K1 = math.sqrt(p) * Z

    return [K0, K1]

# ============================================================
# Time-based noise padding
# ============================================================
def is_only_noise_op(name: str) -> bool:
    return name in {"T1T2_NOISE", "T1_NOISE", "T2_NOISE"}

def op_time(name: str, gate_durations: Dict[str, float]) -> float:
    return gate_durations.get(name, 0.0)


def add_time_based_noise(
    circuit: List[Tuple[str, List[int], float | None]],
    num_qubits: int,
    T1: float,
    T2: float,
    gate_durations: Dict[str, float],
) -> List[Tuple]:
    """
    Insert T1/T2 idle noise between gates based on timing.
    """
    accounted_for_time = [0.0] * num_qubits
    noisy_circuit: List[Tuple] = []

    for op in circuit:
        name = op[0]
        acted_on = op[1]
        # skip adding noise between pure-noise ops
        if is_only_noise_op(name):
            noisy_circuit.append(op)
            continue

        time_to_elapse = op_time(name, gate_durations)

        # If op acts on multiple qubits, bring them to the same time
        if len(acted_on) > 1:
            max_time = max(accounted_for_time[q] for q in acted_on)


        # Synchronize multi-qubit gates
        if len(acted_on) > 1:
            max_time = max(accounted_for_time[q] for q in acted_on)
            for q in acted_on:
                if accounted_for_time[q] < max_time:
                    idle = max_time - accounted_for_time[q]
                    λ1, λ2 = thermal_relaxation_error_rate(T1, T2, idle)
                    if λ1 > 0.0 or λ2 > 0.0:
                        noisy_circuit.append(
                            ("T1T2_NOISE", [q], λ1, λ2, idle)
                        )
                    accounted_for_time[q] = max_time

        noisy_circuit.append(op)

        for q in acted_on:
            accounted_for_time[q] += time_to_elapse

    # Pad final idle time
    end_time = max(accounted_for_time)
    for q in range(num_qubits):
        if accounted_for_time[q] < end_time:
            idle = end_time - accounted_for_time[q]
            λ1, λ2 = thermal_relaxation_error_rate(T1, T2, idle)
            if λ1 > 0.0 or λ2 > 0.0:
                noisy_circuit.append(
                    ("T1T2_NOISE", [q], λ1, λ2, idle)
                )

    return noisy_circuit

# ============================================================
# Kraus helpers for embedding
# ============================================================
def make_operations_probs_from_kraus(
    kraus_ops: List[torch.Tensor]
) -> List[Tuple[torch.Tensor, float]]:
    """
    Convert Kraus ops {K_i} into (U_i, p_i) form
    suitable for kraus_operator().
    """
    L = len(kraus_ops)
    if L == 0:
        return [(torch.eye(2, dtype=torch.cfloat), 1.0)]

    p = 1.0 / L
    scale = math.sqrt(L)
    return [(scale * K, p) for K in kraus_ops]

def embed_single_qubit_kraus(
    kraus_ops: List[torch.Tensor],
    qubit: int,
    num_qubits: int,
) -> List[torch.Tensor]:
    """
    Embed single-qubit Kraus operators into full 2^n space.
    """
    full_ops: List[torch.Tensor] = []

    for K in kraus_ops:
        op = None
        for i in range(num_qubits):
            factor = K if i == qubit else I2
            op = factor if op is None else torch.kron(op, factor)
        full_ops.append(op)

    return full_ops
