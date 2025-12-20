from typing import List, Tuple, Dict
import math
import torch

# ============================================================
# Notes on Realistic Superconducting Transmon Parameters
# (Based on contemporary fixed-frequency transmon literature)
# Using Manenti-Motta Textbook, wikipedia, Blais et al.  Circuit QED, Chen Wang's Guide on Superconducting devices 
# ============================================================
# 
# UNIT CONVENTION: All times throughout this module are in MICROSECONDS (μs).
# This includes: T1, T2, gate_durations, idle_time, and all internal calculations.
# Conversion: 1 ns = 0.001 μs, 1 μs = 1000 ns
# 
# T1 (Energy Relaxation Time):
#   - Typical range: 50–120 μs
#   - State-of-art: 70–100 μs (representative average)
#   - Optimized (tantalum, 3D): up to 300+ μs
# 
# T2 (Overall Coherence Time):
#   - Typical range: 50–200 μs
#   - Relationship: T2 ~ T1 to 2T1
#   - Representative: 70–150 μs
# 
# T_φ (Pure Dephasing Time):
#   - Derived from: 1/T2 = 1/(2T1) + 1/T_φ
#   - Typically: T_φ ≥ T1 for well-engineered devices
#   - Representative: 100+ μs
# 
# Gate Times (Microwave-based, all-software Z rotations, converted to μs):
#   - Single-qubit rotations (X, Y, H): 20–30 ns = 0.020–0.030 μs
#   - Z rotations: 0 ns = 0.0 μs (virtual gates, frame update only)
#   - CNOT (microwave cross-resonance): 100–300 ns = 0.1–0.3 μs
#   - CNOT (flux-pulsed CZ + locals): 30–50 ns = 0.03–0.05 μs + overheads
# 
# ============================================================

# Identity
I2 = torch.eye(2, dtype=torch.cfloat)

# ============================================================
# Helper: clamp probabilities to [0, 1]
# ============================================================
def clamp_prob(p: float) -> float:
    return max(0.0, min(1.0, p))

# ============================================================
# Thermal relaxation error rates (T1 / T2 / Tφ)
# ============================================================
def compute_tphi_from_t2(T1: float, T2: float) -> float:
    """
    Compute pure dephasing time T_φ from T1 and T2.
    
    Physics: 1/T2 = 1/(2T1) + 1/T_φ
    Therefore: T_φ = 1 / (1/T2 - 1/(2T1)) = T2*T1 / (2*T1 - T2)
    
    For realistic transmons:
    - T1 ≈ 70–100 μs (energy relaxation)
    - T2 ≈ 70–200 μs (overall coherence time)
    - T_φ ≈ 100+ μs (pure dephasing, often T_φ ≥ T1)
    """
    if T1 <= 0 or T2 <= 0:
        return float('inf')
    
    denominator = 2 * T1 - T2
    if abs(denominator) < 1e-12:  # T2 ≈ 2*T1, T_φ → ∞
        return float('inf')
    if denominator <= 0:  # Unphysical: T2 > 2*T1
        return float('inf')
    
    return T2 * T1 / denominator


def thermal_relaxation_error_rate(
    T1: float,
    T2: float,
    idle_time: float
) -> tuple[float, float]:
    """
    Compute dimensionless probabilities (λ1, λ_φ) for T1/T2 decoherence.
    
    Physics: 1/T2 = 1/(2T1) + 1/T_φ  →  T_φ is implicit in T1 and T2.
    
    UNITS: All parameters (T1, T2, idle_time) MUST be in MICROSECONDS (μs).
    Returns dimensionless probabilities λ1, λ_φ ∈ [0,1].
    
    Typical values (in μs):
    - T1: 70–100 μs (energy relaxation)
    - T2: 70–150 μs (coherence time, constraint: T2 ≤ 2*T1)
    - gate_durations: 0.025 μs (single-qubit), 0.2 μs (CNOT)
    """
    # Validate physical constraints
    assert T1 > 0, f"T1 must be positive, got {T1}"
    assert T2 > 0, f"T2 must be positive, got {T2}"
    
    if idle_time <= 0.0:
        return 0.0, 0.0

    # λ1: dimensionless probability of amplitude damping during idle_time
    λ1 = 1.0 - math.exp(-idle_time / T1)
    
    # λ_φ: dimensionless probability of pure dephasing during idle_time
    # Dephasing rate (1/time): 1/T_φ = 1/T2 - 1/(2T1)
    dephasing_rate = (1.0 / T2) - (1.0 / (2.0 * T1))
    λ_φ = 0.5 * (1.0 - math.exp(-idle_time * dephasing_rate))

    return clamp_prob(λ1), clamp_prob(λ_φ)

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
def phase_damping_kraus(λ_φ: float) -> List[torch.Tensor]:
    """
    Single-qubit pure dephasing channel (T_φ decoherence).
    
    Parameter λ_φ is a dimensionless probability ∈ [0,1], already integrated
    over some time interval from physical T_φ via thermal_relaxation_error_rate.
    
    Channel: ρ ↦ (1-λ_φ)ρ + λ_φ Z ρ Z, so ρ_01 → (1-2λ_φ)ρ_01 per application.
    """
    λ_φ = clamp_prob(λ_φ)

    if λ_φ == 0.0:
        return [torch.eye(2, dtype=torch.cfloat)]

    p = λ_φ
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
    gate_noise_fraction: float = 1.0,
) -> List[Tuple]:
    """
    Insert T1/T2 idle noise between gates based on timing.
    
    Parameters:
        circuit: List of gate operations [(gate_name, [qubits], param), ...]
        num_qubits: Number of qubits in the circuit
        T1: Amplitude damping time constant (μs)
        T2: Dephasing time constant (μs)
        gate_durations: Map of gate names to their durations (all in μs)
        gate_noise_fraction: Fraction of gate time to add as simulated constant
            decoherence noise (0.0 to 1.0). This simulates continuous T1/T2 
            relaxation that occurs even while gates are being applied, by 
            inserting noise "gaps" before each gate proportional to its duration.
            Default is 0.0 (only idle noise, no gate-time decoherence).
    
    UNITS: All time quantities (T1, T2, gate_durations) must be in MICROSECONDS (μs).
    
    Returns:
        Extended circuit with T1T2_NOISE operations inserted.
    """
    gate_noise_fraction = max(0.0, min(1.0, gate_noise_fraction))  # clamp to [0, 1]
    
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

        # Add simulated constant noise (decoherence during gate execution)
        # This models T1/T2 relaxation that occurs even while a gate is applied
        if gate_noise_fraction > 0.0 and time_to_elapse > 0.0:
            noise_time = time_to_elapse * gate_noise_fraction
            for q in acted_on:
                λ1, λ2 = thermal_relaxation_error_rate(T1, T2, noise_time)
                if λ1 > 0.0 or λ2 > 0.0:
                    noisy_circuit.append(
                        ("T1T2_NOISE", [q], λ1, λ2, noise_time)
                    )

        noisy_circuit.append(op)

        for q in acted_on:
            accounted_for_time[q] += time_to_elapse

    # Pad final idle time
    end_time = max(accounted_for_time) if accounted_for_time else 0.0
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
