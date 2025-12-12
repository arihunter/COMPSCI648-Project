import torch
import math

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
def RX(theta):
    return torch.cos(theta/2)*I2 - 1j*torch.sin(theta/2)*X

def RY(theta):
    return torch.cos(theta/2)*I2 - 1j*torch.sin(theta/2)*Y

def RZ(theta):
    return torch.cos(theta/2)*I2 - 1j*torch.sin(theta/2)*Z

# ───────────────────────────────────────────────────────────────
# Two-qubit gates
# ───────────────────────────────────────────────────────────────
CNOT = torch.tensor([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=torch.cfloat)

SWAP = torch.tensor([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]], dtype=torch.cfloat)

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
    gate_axes = list(range(k))

    # tensordot: gate contracts with selected axes of state_tensor
    result = torch.tensordot(gate_tensor, state_tensor, dims=(gate_axes, state_axes))

    # After contraction, the result has indices: (output indices) + (remaining state indices)
    # We need to permute so that output indices go back into proper qubit positions
    # build permutation list
    remaining = [i for i in range(num_qubits) if i not in targets]
    new_order = list(range(k)) + [k + i for i in range(len(remaining))]

    result = result.permute(new_order)

    # reshape back to a vector of length 2^n
    return result.reshape(-1)

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

# ───────────────────────────────────────────────────────────────
# Example usage
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 5

    # random initial state
    state = random_state(n)
    print("Initial general state:", state)

    # Apply a Hadamard on qubit 0
    state = apply_gate(state, H, [0], n)

    # Apply CNOT on qubits 0→1
    state = apply_gate(state, CNOT, [0,1], n)

    print("State after gates:", state)
    print("Measurement samples:", measure(state))
