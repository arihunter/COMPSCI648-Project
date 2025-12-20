from PIL.Image import new
from locale import str
import torch
import math
from constants import DEFAULT_GATE_DURATIONS, DEFAULT_T1, DEFAULT_T2
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
    if isinstance(amplitudes, torch.Tensor):
        state = amplitudes.to(dtype=torch.cfloat).clone().detach()
    else:
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
    
    # Vectorized batched approach: compute all U @ ρ @ U† simultaneously
    # Stack operators and probabilities for batch processing
    operators = torch.stack([op[0] for op in operations_probs])  # (n_ops, dim, dim)
    probs = torch.tensor([op[1] for op in operations_probs], dtype=density.dtype, device=density.device)  # (n_ops,)
    
    # Compute adjoints once
    operators_conj_t = torch.adjoint(operators)  # (n_ops, dim, dim)
    
    # Embarrassingly parallel: batch matrix multiply
    # Step 1: U @ ρ for all operators simultaneously
    temp = torch.matmul(operators, density)  # (n_ops, dim, dim)
    
    # Step 2: (U @ ρ) @ U† for all operators simultaneously
    weighted_results = torch.matmul(temp, operators_conj_t)  # (n_ops, dim, dim)
    
    # Step 3: weight by probabilities and sum
    # Reshape probs to broadcast correctly: (n_ops, 1, 1) for multiplication with (n_ops, dim, dim)
    weighted_results = weighted_results * probs[:, None, None]
    
    return weighted_results.sum(dim=0)

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
# Optimized density gate application (avoids building full unitaries)
# ───────────────────────────────────────────────────────────────

def apply_single_qubit_gate_density(
    density: torch.Tensor,
    U: torch.Tensor,
    q: int,
    num_qubits: int,
) -> torch.Tensor:
    """
    Apply single-qubit unitary U to qubit q on density matrix.
    # ρ' = (U ⊗ I) ρ (U† ⊗ I), implemented via tensor contractions.
    # """
    # # Cache permutation orders to reduce Python overhead per gate application
    # key = (num_qubits, q)
    # if not hasattr(apply_single_qubit_gate_density, "_perm_cache"):
    #     apply_single_qubit_gate_density._perm_cache = {}
    # perm_cache = apply_single_qubit_gate_density._perm_cache

    # if key in perm_cache:
    #     row_order, final_order = perm_cache[key]
    # else:
    #     remaining_rows = [i for i in range(num_qubits) if i != q]
    #     row_order = []
    #     for i in range(num_qubits):
    #         if i == q:
    #             row_order.append(0)  # U_out
    #         else:
    #             idx = remaining_rows.index(i)
    #             row_order.append(1 + idx)

    #     remaining_cols = [i for i in range(num_qubits) if i != q]
    #     final_order = list(range(1, 1 + num_qubits))  # rows stay in order
    #     for i in range(num_qubits):
    #         if i == q:
    #             final_order.append(0)  # Uc_out
    #         else:
    #             idx = remaining_cols.index(i)
    #             final_order.append(1 + num_qubits + idx)

    #     perm_cache[key] = (row_order, final_order)

    # # Reshape density into (row qubits, column qubits)
    # ρ = density.reshape([2]*num_qubits + [2]*num_qubits)

    # # Left multiply on row axis q: contract U(in) with row[q]
    # tmp = torch.tensordot(U, ρ, dims=([1], [q]))  # U_out + row_rem + col
    # tmp = tmp.permute(row_order + list(range(num_qubits, num_qubits + num_qubits)))

    # # Right multiply on column axis q
    # U_dag = U.conj().transpose(0, 1)
    # tmp2 = torch.tensordot(U_dag, tmp, dims=([1], [num_qubits + q]))  # Uc_out + row + col_rem

    # ρ_final = tmp2.permute(final_order).reshape(2**num_qubits, 2**num_qubits)
    # return ρ_final
    # ρ' = (U ⊗ I) ρ (U† ⊗ I), via full unitary embedding for correctness.
    # Build full unitary and apply via Kraus (known working path)
    U_full = build_full_unitary(U, [q], num_qubits)
    return kraus_operator(density, [(U_full, 1.0)])



def apply_two_qubit_gate_density(
    density: torch.Tensor,
    U2: torch.Tensor,
    q0: int,
    q1: int,
    num_qubits: int,
) -> torch.Tensor:
    """
    Apply two-qubit unitary U2 to qubits (q0, q1) on density matrix via contractions.
    ρ' = (U2 ⊗ I) ρ (U2† ⊗ I)
    """
    # Ensure ordered targets for consistent placement
    targets = [q0, q1]
    k = 2
    ρ = density.reshape([2]*num_qubits + [2]*num_qubits)

    # Cache permutation orders per (num_qubits, targets)
    key = (num_qubits, tuple(sorted(targets)))
    if not hasattr(apply_two_qubit_gate_density, "_perm_cache"):
        apply_two_qubit_gate_density._perm_cache = {}
    perm_cache = apply_two_qubit_gate_density._perm_cache

    if key in perm_cache:
        row_order, final_order = perm_cache[key]
    else:
        remaining_rows = [i for i in range(num_qubits) if i not in targets]
        row_order = []
        for i in range(num_qubits):
            if i in targets:
                row_order.append(targets.index(i))  # 0 or 1 from out axes
            else:
                row_order.append(k + remaining_rows.index(i))

        remaining_cols = [i for i in range(num_qubits) if i not in targets]
        final_order = list(range(k, k + num_qubits))  # rows stay in order
        for i in range(num_qubits):
            if i in targets:
                final_order.append(targets.index(i))  # outc axes positions
            else:
                final_order.append(k + num_qubits + remaining_cols.index(i))

        perm_cache[key] = (row_order, final_order)

    # Left multiply: contract U2(in axes) with row axes at targets
    G = U2.reshape([2]*k + [2]*k)
    tmp = torch.tensordot(G, ρ, dims=(list(range(k, 2*k)), targets))
    tmp = tmp.permute(row_order + list(range(k + len([i for i in range(num_qubits) if i not in targets]), k + len([i for i in range(num_qubits) if i not in targets]) + num_qubits)))

    # Right multiply on column axes at targets
    G_dag = U2.conj().transpose(0, 1).reshape([2]*k + [2]*k)
    col_targets = [num_qubits + t for t in targets]
    tmp2 = torch.tensordot(G_dag, tmp, dims=(list(range(k, 2*k)), col_targets))

    ρ_final = tmp2.permute(final_order).reshape(2**num_qubits, 2**num_qubits)
    return ρ_final


# ───────────────────────────────────────────────────────────────
# Apply a named gate to a density matrix (optimized path)
# ───────────────────────────────────────────────────────────────

def apply_named_gate_density(
    density: torch.Tensor,
    op: Tuple[str, List[int], float | None],
    num_qubits: int,
    unitary_cache: Dict[Tuple[str, Tuple[int, ...], float | None], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Optimized density update:
    - Single-qubit gates: einsum/tensordot contractions (no full unitary)
    - Two-qubit gates (e.g., CNOT): contraction on two axes
    Falls back to full-unitary embedding only if gate arity > 2.
    """
    gate_name = op[0]
    qubits = op[1]
    is_param = gate_name in PARAMETRIC_GATES
    param = op[2] if is_param else None

    # Resolve unitary matrix for gate
    U = PARAMETRIC_GATES[gate_name](param) if is_param else GATE_LIBRARY[gate_name]

    if len(qubits) == 1:
        return apply_single_qubit_gate_density(density, U, qubits[0], num_qubits)
    elif len(qubits) == 2:
        return apply_two_qubit_gate_density(density, U, qubits[0], qubits[1], num_qubits)
    else:
        # Rare case: fall back to full-unitary path (cached)
        key = (gate_name, tuple(qubits), param)
        if unitary_cache is not None and key in unitary_cache:
            U_full = unitary_cache[key]
        else:
            U_full = build_full_unitary(U, qubits, num_qubits)
            if unitary_cache is not None:
                unitary_cache[key] = U_full
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
    Apply T1 (amplitude damping) and T_φ (pure dephasing) channels.
    noise_op = ("T1T2_NOISE", [q], λ1, λ_φ, idle_time)
    
    Applies: (1) amplitude damping, then (2) pure dephasing.
    Keeps the map trace-preserving.
    """
    _, qubits, λ1, λ_φ, _ = noise_op
    q = qubits[0]

    # Caches to avoid rebuilding kron-embedded Kraus ops for identical parameters
    # Keys include λ values, target qubit, and num_qubits to ensure correctness
    if not hasattr(apply_T1T2_noise_op, "_amp_cache"):
        apply_T1T2_noise_op._amp_cache = {}
        apply_T1T2_noise_op._deph_cache = {}
    amp_cache = apply_T1T2_noise_op._amp_cache
    deph_cache = apply_T1T2_noise_op._deph_cache

    # 1) Amplitude damping (T1: |1⟩ → |0⟩ relaxation)
    if λ1 > 0.0:
        amp_key = (λ1, q, num_qubits)
        if amp_key in amp_cache:
            amp_ops_probs = amp_cache[amp_key]
        else:
            amp_kraus = amplitude_damping_kraus(λ1)
            amp_full  = embed_single_qubit_kraus(amp_kraus, q, num_qubits)
            amp_ops_probs = make_operations_probs_from_kraus(amp_full)
            amp_cache[amp_key] = amp_ops_probs
        density = kraus_operator(density, amp_ops_probs)

    # 2) Pure dephasing (T_φ: coherence loss from 1/T2 - 1/(2T1))
    if λ_φ > 0.0:
        deph_key = (λ_φ, q, num_qubits)
        if deph_key in deph_cache:
            deph_ops_probs = deph_cache[deph_key]
        else:
            deph_kraus = phase_damping_kraus(λ_φ)
            deph_full  = embed_single_qubit_kraus(deph_kraus, q, num_qubits)
            deph_ops_probs = make_operations_probs_from_kraus(deph_full)
            deph_cache[deph_key] = deph_ops_probs
        density = kraus_operator(density, deph_ops_probs)

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
    gate_durations: Dict[str, float]| None = None,
    gate_noise_fraction: float = 1.0,
) -> torch.Tensor:
    """
    Execute a circuit with T1/T2 time-based idle noise, in the density-matrix picture.

    UNITS: All time quantities (T1, T2, gate_durations) must be in MICROSECONDS (μs).

    Inputs:
        initial_state : state vector (length 2^n)
        circuit       : [(gate_name, [qubits]), ...]
        num_qubits    : number of qubits
        T1, T2        : relaxation times (μs)
        gate_durations: map gate_name -> duration (μs)
        gate_noise_fraction: fraction of gate time to add as simulated constant
            decoherence noise (0.0 to 1.0). Models T1/T2 relaxation during gates.

    Output:
        density matrix ρ_final (2^n x 2^n)
    """
    if gate_durations is None:
        gate_durations = dict(DEFAULT_GATE_DURATIONS)
        
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
            gate_noise_fraction=gate_noise_fraction,
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

    # Realistic gate durations (all times in μs) for superconducting transmons
    gate_durations = dict(DEFAULT_GATE_DURATIONS)

    T1 = DEFAULT_T1  # μs
    T2 = 120.0  # μs

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

