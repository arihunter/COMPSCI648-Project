from typing import List, Tuple, Dict
import math
import torch

I2 = torch.eye(2, dtype=torch.cfloat) 

# ───────────────────────────────────────────────────────────────
# Kraus helpers for T1 / T2
# ───────────────────────────────────────────────────────────────

def thermal_relaxation_error_rate(T1: float, T2: float, idle_time: float) -> tuple[float, float]:
    """
    Given T1, T2 and an idle time Δt, return (λ1, λ2):

    - λ1: amplitude damping parameter (T1), probability of |1> decaying.
          λ1 = 1 - exp(-Δt / T1)
    - λ2: pure dephasing parameter chosen so that total coherence decays with T2.
          exp(-Δt/T2) = exp(-Δt/(2T1)) (1 - 2 λ2)
          ⇒ λ2 = (1 - exp(-Δt/T2 + Δt/(2T1))) / 2
    """
    if idle_time <= 0.0:
        return 0.0, 0.0

    # Amplitude damping part (T1)
    if T1 <= 0 or math.isinf(T1):
        λ1 = 0.0
    else:
        λ1 = 1.0 - math.exp(-idle_time / T1)

    # Pure dephasing part, adjusted so T2 is the net coherence time
    if T2 <= 0 or math.isinf(T2):
        λ2 = 0.0
    else:
        if T1 <= 0 or math.isinf(T1):
            # No T1 contribution; T2 is purely dephasing
            λ2 = (1.0 - math.exp(-idle_time / T2)) / 2.0
        else:
            # Use the standard relation:
            # λ2 = (1 - exp(-Δt/T2 + Δt/(2T1))) / 2
            exp_arg = -idle_time / T2 + idle_time / (2.0 * T1)
            λ2 = (1.0 - math.exp(exp_arg)) / 2.0 
            # Guard against small negative numerical noise
            λ2 = max(0.0, min(1.0, λ2))

    return λ1, λ2


def amplitude_damping_kraus(λ1: float) -> List[torch.Tensor]:
    """
    Single-qubit amplitude damping channel with parameter λ1.
    Population in |1> decays to |0> with probability λ1.
    Nielson&Chuang eq 8.108
    """
    if λ1 <= 0.0:
        return [torch.eye(2, dtype=torch.cfloat)]

    p = λ1
    
    K0 = torch.tensor([[1.0, 0.0],
                       [0.0, math.sqrt(1.0 - p)]], dtype=torch.cfloat)
    K1 = torch.tensor([[0.0, math.sqrt(p)],
                       [0.0, 0.0]], dtype=torch.cfloat)
    return [K0, K1]

#TODO:
def phase_damping_kraus(λ2: float) -> List[torch.Tensor]:
    """
    Single-qubit dephasing channel with parameter λ2.
    Off-diagonals get multiplied by (1 - 2 λ2).
    """
    if λ2 <= 0.0:
        return [torch.eye(2, dtype=torch.cfloat)]

    p = λ2
    # TODO: review this math

    I2 = torch.eye(2, dtype=torch.cfloat)
    K0 = math.sqrt(1.0 - p) * I2
    K1 = math.sqrt(p) * torch.tensor([[1.0, 0.0],
                       [0.0, -1.0]], dtype=torch.cfloat)
    return [K0, K1]

# ───────────────────────────────────────────────────────────────
# Time-based noise padding (circuit → noisy_circuit)
# ───────────────────────────────────────────────────────────────

def is_only_noise_op(name: str) -> bool:
    # extend this set if you introduce other pure-noise ops
    return name in {"T1T2_NOISE", "T1_NOISE", "T2_NOISE"}

def op_time(name: str, gate_durations: Dict[str, float]) -> float:
    """
    Look up how long a gate takes. Defaults to 0.0 if unknown.
    """
    return gate_durations.get(name, 0.0)


def add_time_based_noise(
    circuit: List[Tuple[str, List[int], float | None]],
    num_qubits: int,
    T1: float,
    T2: float,
    gate_durations: Dict[str, float],
) -> List[Tuple]:
    """
    Mirror of the Julia noisify circuit for T1T2 noise from QEPOptimize, but in Python.

    Input:
        circuit        : [(gate_name, [qubits]), ...]
        num_qubits     : total number of qubits in the register
        T1, T2         : relaxation times
        gate_durations : map gate_name -> duration

    Output:
        noisy_circuit: list containing:
          - (gate_name, [qubits])  for physical gates
          - ("T1T2_NOISE", [q], λ1, λ2, idle_time) for idle noise
    """
    assert num_qubits > 0
    accounted_for_time = [0.0] * num_qubits
    noisy_circuit: List[Tuple] = []

    # main loop over operations
    for op in circuit:
        name = op[0]
        acted_on_qubits = op[1]
        # skip adding noise between pure-noise ops
        if is_only_noise_op(name):
            noisy_circuit.append(op)
            continue

        time_to_elapse = op_time(name, gate_durations)

        # If op acts on multiple qubits, bring them to the same time
        if len(acted_on_qubits) > 1:
            max_time = max(accounted_for_time[q] for q in acted_on_qubits)

            for q in acted_on_qubits:
                if accounted_for_time[q] < max_time:
                    idle_time = max_time - accounted_for_time[q]
                    λ1, λ2 = thermal_relaxation_error_rate(T1, T2, idle_time)

                    if λ1 > 0.0 or λ2 > 0.0:
                        noisy_circuit.append(
                            ("T1T2_NOISE", [q], λ1, λ2, idle_time)
                        )

                    accounted_for_time[q] = max_time

        # Apply the actual gate
        noisy_circuit.append(op)

        # Update times for acted qubits
        for q in acted_on_qubits:
            accounted_for_time[q] += time_to_elapse

    # After all operations, pad remaining idle time up to the circuit end
    end_time = max(accounted_for_time)

    for q in range(num_qubits):
        if accounted_for_time[q] < end_time:
            idle_time = end_time - accounted_for_time[q]
            λ1, λ2 = thermal_relaxation_error_rate(T1, T2, idle_time)

            if λ1 > 0.0 or λ2 > 0.0:
                noisy_circuit.append(
                    ("T1T2_NOISE", [q], λ1, λ2, idle_time)
                )

    return noisy_circuit

# To scale error operators so they fit into the main kraus op
def make_operations_probs_from_kraus(kraus_ops: List[torch.Tensor]) -> List[tuple[torch.Tensor, float]]:
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
    Return {K_full} where each K_full acts on the full 2^n space,
    with K applied on 'qubit' and identity elsewhere.
    """
    full_ops: List[torch.Tensor] = []
    for K in kraus_ops:
        op = None
        for i in range(num_qubits):
            factor = K if i == qubit else I2
            op = factor if op is None else torch.kron(op, factor)
        full_ops.append(op)
    return full_ops
