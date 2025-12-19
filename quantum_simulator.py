from PIL.Image import new
from locale import str
import torch
import math
from error_kraus import *
from visualizer import plot_measurement_comparison
# ───────────────────────────────────────────────────────────────
# State utilities
# ───────────────────────────────────────────────────────────────
def zero_state(num_qubits: int):
    state = torch.zeros(2**num_qubits, dtype=torch.cfloat)
    state[0] = torch.complex(torch.tensor(1.0), torch.tensor(0.0))
    return state

def random_state(num_qubits: int):
    real = torch.randn(2**num_qubits)
    imag = torch.randn(2**num_qubits)
    state = torch.complex(real, imag)
    return state / torch.linalg.norm(state)

def custom_state(amplitudes):
    state = torch.tensor(amplitudes, dtype=torch.cfloat)
    return state / torch.linalg.norm(state)

# ───────────────────────────────────────────────────────────────
# One-qubit gates
# ───────────────────────────────────────────────────────────────
X = torch.tensor([[0, 1],
                  [1, 0]], dtype=torch.cfloat)

Y = torch.tensor([[0, -1j],
                  [1j, 0]], dtype=torch.cfloat)

Z = torch.tensor([[1,  0],
                  [0, -1]], dtype=torch.cfloat)

H = (1/math.sqrt(2)) * torch.tensor([[1,  1],
                                     [1, -1]], dtype=torch.cfloat)

S = torch.tensor([[1, 0],
                  [0, 1j]], dtype=torch.cfloat)

theta_t = math.pi/4
phase = torch.exp(torch.complex(torch.tensor(0.0), torch.tensor(theta_t)))
T = torch.tensor([[1, 0],
                  [0, phase]], dtype=torch.cfloat)

I2 = torch.eye(2, dtype=torch.cfloat) 

# ───────────────────────────────────────────────────────────────
# Parameterized rotations
# ───────────────────────────────────────────────────────────────
def RX(theta: float):
    theta = theta if torch.is_tensor(theta) else torch.tensor(theta)
    return torch.cos(theta/2)*I2 - 1j*torch.sin(theta/2)*X

def RY(theta: float):
    theta = theta if torch.is_tensor(theta) else torch.tensor(theta)
    return torch.cos(theta/2)*I2 - 1j*torch.sin(theta/2)*Y

def RZ(theta: float):
    theta = theta if torch.is_tensor(theta) else torch.tensor(theta)
    return torch.cos(theta/2)*I2 - 1j*torch.sin(theta/2)*Z

# ───────────────────────────────────────────────────────────────
# Two-qubit gates
# ───────────────────────────────────────────────────────────────
CNOT = torch.tensor([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=torch.cfloat)

# For testing "NOTC" -> controlled not with first qubit as target and second as control 
NOTC = torch.tensor([[1,0,0,0],
                     [0,0,0,1],
                     [0,0,1,0],
                     [0,1,0,0]], dtype=torch.cfloat)


SWAP = torch.tensor([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]], dtype=torch.cfloat)

# ───────────────────────────────────────────────────────────────
# Gate library (string name → matrix)
# extend as needed
# ───────────────────────────────────────────────────────────────

GATE_LIBRARY: Dict[str, torch.Tensor] = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "CNOT": CNOT,
    "NOTC": NOTC,
    "SWAP": SWAP
}

# PARAMETRIC_GATES: Dict[str, callable[[float], torch.Tensor]] = {
# PARAMETRIC_GATES: Dict[str, Unknown, torch.Tensor] = {
PARAMETRIC_GATES = {
    "RZ": RZ,
    "RY": RY, 
    "RX": RX
}

# ───────────────────────────────────────────────────────────────
# Apply a gate using tensor reshape + contraction
# ───────────────────────────────────────────────────────────────
def apply_gate(state: torch.Tensor, gate: torch.Tensor, targets: list, num_qubits: int):
    """
    Apply 'gate' to qubits in 'targets' on an n-qubit state vector.
    Uses tensor reshape + tensordot contraction.
    """
    # reshape state to tensor of shape (2,2,...,2) with one axis per qubit
    state_tensor = state.reshape([2]*num_qubits)

    # number of qubits the gate acts on
    k = len(targets)

    # reshape gate to shape (2,...,2,2,...,2): (input indices, output indices)
    new_shape = [2]*k + [2]*k
    gate_tensor = gate.reshape(new_shape)

    # calculate the axes for tensordot: contract gate input indices with state tensor indices
    state_axes = targets
    # Select the input axes of the reshaped gate tensor (last k axes) to contract with state,
    # ensuring correct matrix-vector multiplication: gate @ state
    gate_axes = list(range(k, 2*k))  

    # tensordot: gate contracts with selected axes of state_tensor
    result = torch.tensordot(gate_tensor, state_tensor, dims=(gate_axes, state_axes))

    # After contraction, the result has indices: (output indices) + (remaining state indices)
    # We need to permute so that output indices go back into proper qubit positions
    # Fix: Build permutation to correctly place output axes at target qubit positions
    remaining = [i for i in range(num_qubits) if i not in targets]

    new_order = []
    for i in range(num_qubits):
        if i in targets:
            idx = targets.index(i)
            new_order.append(idx)
        else:
            idx = remaining.index(i)
            new_order.append(k + idx)

    result = result.permute(new_order)

    # reshape back to a vector of length 2^n
    return result.reshape(-1)


"""Apply circuit list (in the style of circuits fed to "run_noisy_circuit_density") to a state, but stay in ket/vector notation

Example circuit:
    circuit = [
        ("H",   [0]),
        ("CNOT",[0,1]),
        (RY,   [2], math.pi/4),  # parametric gate
    ]
    
EX: 
state = zero_state(3)
new_state = apply_circuit_to_ket(state, circuit, 3)
"""
def apply_circuit_to_ket(state: torch.Tensor, circuit: List[Tuple[str, List[int]]], num_qubits: int):
    new_state = state.clone()
    for op in circuit:
        name = op[0]
        qubits = op[1]
        if name in PARAMETRIC_GATES:
            param = op[2]
            gate = PARAMETRIC_GATES[name](param)
        else:
            gate = GATE_LIBRARY[name]
        new_state = apply_gate(new_state, gate, qubits, num_qubits)
    return new_state
# ───────────────────────────────────────────────────────────────
# Measurement
# ───────────────────────────────────────────────────────────────
def measure(state: torch.Tensor, shots=1024):
    probs = torch.abs(state)**2
    outcomes = torch.multinomial(probs, shots, replacement=True)
    counts = {}
    for o in outcomes.tolist():
        bitstr = format(o, f"0{int(math.log2(len(state)))}b")
        counts[bitstr] = counts.get(bitstr, 0) + 1
    return counts

def measure_kraus(density: torch.Tensor, shots=1024):
    # should be no imaginary values on the density matrix
    assert abs(sum(torch.tensor([density[i][i].imag for i in range(len(density))]))) < 10e-10
    probs = torch.tensor([density[i][i].real for i in range(len(density))])
    outcomes = torch.multinomial(probs, shots, replacement=True)
    counts = {}
    for o in outcomes.tolist():
        bitstr = format(o, f"0{int(math.log2(len(state)))}b")
        counts[bitstr] = counts.get(bitstr, 0) + 1
    return counts
# ───────────────────────────────────────────────────────────────
# Kraus Operator logic
# ───────────────────────────────────────────────────────────────
def state_to_density(state: torch.Tensor):
    # Compute density matrix |ψ⟩⟨ψ| using outer product
    return torch.outer(state, torch.conj(state))

def kraus_operator(density: torch.Tensor, operations_probs: list[tuple[torch.Tensor,float]]) -> torch.Tensor:
    # Given the density matrix for a state, and a list of (op, probability of op)
    # output the resulting density matrix after the total operator is applied
    
    # Probabilities must sum to 1
    assert sum(ops[1] for ops in operations_probs) == 1
    return sum(
        # the probability of this option
        op[1] *
        # U * ρ * U† (matrix multiplication)
        op[0] @ density @ torch.adjoint(op[0])
        
        # for each op and prob
        for op in operations_probs
    )

# ───────────────────────────────────────────────────────────────
# Embed a local gate as a full 2^n x 2^n unitary
# (using your existing apply_gate on basis states)
# ───────────────────────────────────────────────────────────────

def build_full_unitary(
    gate: torch.Tensor,
    targets: List[int],
    num_qubits: int,
) -> torch.Tensor:
    """
    Construct the full 2^n x 2^n unitary for 'gate' acting on 'targets',
    using the same semantics as apply_gate(state, gate, targets, num_qubits).
    """
    dim = 2**num_qubits
    U_full = torch.zeros((dim, dim), dtype=torch.cfloat)

    # Build columns of U_full by acting on basis states
    for basis_index in range(dim):
        basis_state = torch.zeros(dim, dtype=torch.cfloat)
        basis_state[basis_index] = 1.0 + 0.0j

        out_state = apply_gate(basis_state, gate, targets, num_qubits)
        U_full[:, basis_index] = out_state

    return U_full

# ───────────────────────────────────────────────────────────────
# Apply a named gate to a density matrix via Kraus formalism
# ───────────────────────────────────────────────────────────────

def apply_named_gate_density(
    density: torch.Tensor,
    op: Tuple[str, List[int], float | None],
    num_qubits: int,
    unitary_cache: Dict[Tuple[str, Tuple[int, ...], float | None], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    op = (gate_name, [qubits])

    Uses kraus_operator with a single Kraus operator U_full (probability 1).
    Caches U_full per (gate_name, tuple(qubits)) if unitary_cache is provided.
    
    For parametric gates, use 
    op = (gate_name, [qubits], param)
    ie: ("RX", [0], .4)
    """
    gate_name = op[0]
    qubits = op[1]
    
    # take care of parametric case (have to deal with the param value)
    is_param = gate_name in PARAMETRIC_GATES
    param = op[2] if is_param else None
    key = (gate_name, tuple(qubits), param)

    if unitary_cache is not None and key in unitary_cache:
        U_full = unitary_cache[key]
    else:
        gate = PARAMETRIC_GATES[gate_name](param) if is_param else GATE_LIBRARY[gate_name]
        U_full = build_full_unitary(gate, qubits, num_qubits)
        if unitary_cache is not None:
            unitary_cache[key] = U_full

    # Single Kraus operator with probability 1.0
    return kraus_operator(density, [(U_full, 1.0)])

# ───────────────────────────────────────────────────────────────
# Apply one T1/T2 noise op to a density matrix
# ───────────────────────────────────────────────────────────────

def apply_T1T2_noise_op(
    density: torch.Tensor,
    noise_op: tuple,
    num_qubits: int,
) -> torch.Tensor:
    """
    noise_op = ("T1T2_NOISE", [q], λ1, λ2, idle_time)

    Corrected version: applies amplitude damping and phase damping
    as two *composed* channels, not as a single concatenated Kraus set.
    This keeps the map trace-preserving.
    """
    _, qubits, λ1, λ2, _ = noise_op
    q = qubits[0]

    # 1) Amplitude damping (T1 part)
    if λ1 > 0.0:
        amp_kraus = amplitude_damping_kraus(λ1)            # single-qubit {E_i}
        amp_full  = embed_single_qubit_kraus(amp_kraus, q, num_qubits)  # {E_i^full}
        amp_ops_probs = make_operations_probs_from_kraus(amp_full)
        density = kraus_operator(density, amp_ops_probs)   # ρ ← Σ_i E_i ρ E_i†

    # 2) Pure dephasing (Tφ part)
    if λ2 > 0.0:
        deph_kraus = phase_damping_kraus(λ2)               # single-qubit {F_j}
        deph_full  = embed_single_qubit_kraus(deph_kraus, q, num_qubits)
        deph_ops_probs = make_operations_probs_from_kraus(deph_full)
        density = kraus_operator(density, deph_ops_probs)  # ρ ← Σ_j F_j ρ F_j†

    return density

# ───────────────────────────────────────────────────────────────
# Top-level executor: state → noisy density matrix
# ───────────────────────────────────────────────────────────────

def run_noisy_circuit_density(
    initial_state: torch.Tensor,
    circuit: List[Tuple[str, List[int]]],
    num_qubits: int,
    T1: float,
    T2: float,
    gate_durations: Dict[str, float],
) -> torch.Tensor:
    """
    Execute a circuit with T1/T2 time-based idle noise, in the density-matrix picture.

    Inputs:
        initial_state : state vector (length 2^n)
        circuit       : [(gate_name, [qubits]), ...]
        num_qubits    : number of qubits
        T1, T2        : relaxation times
        gate_durations: map gate_name -> duration

    Output:
        density matrix ρ_final (2^n x 2^n)
    """
    # Convert to density matrix
    density = state_to_density(initial_state)

    # Insert T1/T2 noise ops according to timing logic
    noisy_circuit = circuit
    if T1 != 0 and T2 != 0:
        noisy_circuit = add_time_based_noise(
            circuit=circuit,
            num_qubits=num_qubits,
            T1=T1,
            T2=T2,
            gate_durations=gate_durations,
        )

    # Optional cache for full unitaries
    unitary_cache: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}

    # Execute noisy circuit
    for op in noisy_circuit:
        name = op[0]

        if name == "T1T2_NOISE":
            density = apply_T1T2_noise_op(density, op, num_qubits)
        else:
            # Physical gate
            density = apply_named_gate_density(
                density, (name, op[1],  op[2] if name in PARAMETRIC_GATES else None), num_qubits, unitary_cache
            )

    return density

# ───────────────────────────────────────────────────────────────
# Example usage
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 5
    SHOTS = 10000

    # random initial state
    init_state = random_state(n)
    # init_state = zero_state(n)

    print("Initial general state:", init_state)

    # Apply a Hadamard on qubit 0
    state = apply_gate(init_state, H, [0], n)

    # Apply CNOT on qubits 0→1
    state = apply_gate(state, CNOT, [0,1], n)

    # print("State after gates:", state)
    state_counts = measure(state,shots=SHOTS)
    print("Measurement samples:", state_counts)

    # Kraus ops example

    # logical circuit: H on 0, CNOT 0→1
    circuit = [
        ("H",   [0]),
        ("CNOT",[0,1]),
    ]

    # simple uniform durations
    gate_durations ={
        "H":    1,
        "CNOT": 1,
    }

    T1 = 10
    T2 = 20

    # compare normal to kraus result
    ρ_final_normal = run_noisy_circuit_density(
        initial_state=init_state,
        circuit=circuit,
        num_qubits=n,
        T1=0,
        T2=0,
        gate_durations=gate_durations,
    )
    normal_kraus_counts = measure_kraus(ρ_final_normal,shots=SHOTS)
    plot_measurement_comparison(state_counts, normal_kraus_counts,
                            title="Normal vs Kraus measurement distributions")


    # compare to errored kraus result
  
    ρ_final = run_noisy_circuit_density(
        initial_state=init_state,
        circuit=circuit,
        num_qubits=n,
        T1=T1,
        T2=T2,
        gate_durations=gate_durations,
    )
    kraus_counts = measure_kraus(ρ_final,shots=SHOTS)
    print("Measurement samples, error kraus:", kraus_counts)
    

    plot_measurement_comparison(state_counts, kraus_counts,
                            title="Normal vs error Kraus measurement distributions")
  
    # Rotation ops test
    
    # circuit = [
    #     ("RX",   [0],.2),
    #     ("CNOT",[0,1]),
    # ]

    # # simple uniform durations
    # gate_durations ={
    #     "H":    1,
    #     "CNOT": 1,
    #     "RX":   1
    # }

    # T1 = 10
    # T2 = 20

    # # compare normal to kraus result
    # ρ_final_normal = run_noisy_circuit_density(
    #     initial_state=init_state,
    #     circuit=circuit,
    #     num_qubits=n,
    #     T1=0,
    #     T2=0,
    #     gate_durations=gate_durations,
    # )
    # normal_kraus_counts = measure_kraus(ρ_final_normal,shots=SHOTS)
    # plot_measurement_comparison(state_counts, normal_kraus_counts,
    #                         title="Normal vs Kraus measurement distributions")

