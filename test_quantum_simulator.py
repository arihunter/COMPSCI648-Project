import unittest
import torch
import math
from quantum_simulator import *

class TestStates(unittest.TestCase):

    def assert_tensors_close(self, actual, expected):
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as e:
            self.fail(f"\nExpected: {expected}\nReceived: {actual}\nOriginal error: {e}")

    def test_zero_state(self):
        state = zero_state(1)
        self.assertEqual(state[0], torch.complex(torch.tensor(1.0), torch.tensor(0.0)))
        self.assertEqual(state[1], torch.complex(torch.tensor(0.0), torch.tensor(0.0)))

    def test_random_state(self):
        state = random_state(2)
        self.assertAlmostEqual(torch.linalg.norm(state).item(), 1.0, places=5)

    def test_custom_state(self):
        amplitudes = [1, 0, 0, 1j]
        state = custom_state(amplitudes)
        norm = torch.linalg.norm(state)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_measure(self):
        state = zero_state(1)
        counts = measure(state, shots=1000)
        self.assertIn('0', counts)
        self.assertNotIn('1', counts)

        state = apply_gate(state, X, [0], 1)
        counts = measure(state, shots=1000)
        self.assertIn('1', counts)
        self.assertNotIn('0', counts)
        
    def test_3q_state(self):
        state = zero_state(3)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[0] = 1  # |000>
        self.assert_tensors_close(state, expected)

class TestGates(unittest.TestCase):

    def assert_tensors_close(self, actual, expected):
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as e:
            self.fail(f"\nExpected: {expected}\nReceived: {actual}\nOriginal error: {e}")

    def test_apply_gate(self): # Identity
        state = zero_state(1)
        new_state = apply_gate(state, I2, [0], 1)
        expected = torch.tensor([1, 0], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_X_gate(self):
        state = zero_state(1)
        new_state = apply_gate(state, X, [0], 1)
        expected = torch.tensor([0, 1], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    # y gate application must be wrong
    def test_Y_gate(self):
        state = zero_state(1);  # |0>,
        new_state = apply_gate(state, Y, [0], 1) # Y|0> = i|1>
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)
        
        state = zero_state(1); state[0] = 0; state[1] = 1;  # |1>,
        new_state = apply_gate(state, Y, [0], 1) # Y|1> = -i|0>
        expected = torch.tensor([-1j, 0], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_Z_gate(self):
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, Z, [0], 1)
        expected = torch.tensor([0, -1], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_H_gate(self):
        state = zero_state(1) # |0>
        new_state = apply_gate(state, H, [0], 1)
        expected = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_S_gate(self):
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, S, [0], 1)
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_T_gate(self):
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, T, [0], 1)
        phase = torch.exp(torch.complex(torch.tensor(0.0), torch.tensor(math.pi/4)))
        expected = torch.tensor([0, phase], dtype=torch.cfloat)
        self.assert_tensors_close(new_state, expected)

    def test_RX_gate(self):
        theta = math.pi
        gate = RX(theta)
        state = zero_state(1)
        new_state = apply_gate(state, gate, [0], 1)
        expected = torch.tensor([0, -1j], dtype=torch.cfloat)  # RX(pi) |0> = -i |1>
        self.assert_tensors_close(new_state, expected)

    def test_RY_gate(self):
        theta = math.pi
        gate = RY(theta)
        state = zero_state(1)
        new_state = apply_gate(state, gate, [0], 1)
        expected = torch.tensor([0, 1], dtype=torch.cfloat)  # RY(pi) |0> = |1>
        self.assert_tensors_close(new_state, expected)

    def test_RZ_gate(self):
        theta = math.pi
        gate = RZ(theta)
        state = zero_state(1)
        state[0] = 0;state[1] = 1  # |1>
        new_state = apply_gate(state, gate, [0], 1)
        expected = torch.tensor([0, 1j], dtype=torch.cfloat)  # RZ(pi) |1> = i |1>
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_00(self):
        # CNOT |00> = |00>
        state = zero_state(2)
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = zero_state(2)
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_01(self):
        # CNOT |01> = |01>
        state = zero_state(2)
        state[0] = 0;state[1] = 1
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[1] = 1
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_10(self):
        # CNOT |10> = |11>
        state = zero_state(2)
        state[0] = 0; state[2] = 1
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[3] = 1
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_11(self):
        # CNOT |11> = |10>
        state = zero_state(2)
        state[0] = 0;state[3] = 1
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[2] = 1
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_plus_plus(self):
        # CNOT |++> = |++>
        plus = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        state = torch.kron(plus, plus)
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = state
        self.assert_tensors_close(new_state, expected)

    def test_CNOT_gate_plus_minus(self):
        # CNOT |+-> = |-->
        plus = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.cfloat)
        minus = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)], dtype=torch.cfloat)
        state = torch.kron(plus, minus)
        new_state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.kron(minus, minus)
        self.assert_tensors_close(new_state, expected)

    def test_X_Y_Z_anticommute(self):
        state = random_state(1)
        # X and Y anticommute: {X,Y} = 0
        xy = apply_gate(apply_gate(state, X, [0], 1), Y, [0], 1)
        yx = apply_gate(apply_gate(state, Y, [0], 1), X, [0], 1)
        anticommutator = xy + yx
        self.assert_tensors_close(anticommutator, torch.zeros_like(anticommutator))

        # X and Z anticommute: {X,Z} = 0
        xz = apply_gate(apply_gate(state, X, [0], 1), Z, [0], 1)
        zx = apply_gate(apply_gate(state, Z, [0], 1), X, [0], 1)
        anticommutator = xz + zx
        self.assert_tensors_close(anticommutator, torch.zeros_like(anticommutator))

        # Y and Z anticommute: {Y,Z} = 0
        yz = apply_gate(apply_gate(state, Y, [0], 1), Z, [0], 1)
        zy = apply_gate(apply_gate(state, Z, [0], 1), Y, [0], 1)
        anticommutator = yz + zy
        self.assert_tensors_close(anticommutator, torch.zeros_like(anticommutator))

class TestCircuits(unittest.TestCase):
    def assert_tensors_close(self, actual, expected):
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as e:
            self.fail(f"\nExpected: {expected}\nReceived: {actual}\nOriginal error: {e}")



    def test_three_CNOTs_simulate_SWAP(self):
        state = zero_state(2)
        state[0] = 0; state[1] = 1  # |01>
        # Apply CNOT(0,1), CNOT(1,0), CNOT(0,1)
        state = apply_gate(state, CNOT, [0, 1], 2)
        state = apply_gate(state, NOTC, [0, 1], 2)
        state = apply_gate(state, CNOT, [0, 1], 2)
        expected = torch.zeros(4, dtype=torch.cfloat)
        expected[2] = 1  # |10>
        self.assert_tensors_close(state, expected)

    def test_SWAP_with_three_CNOTs(self):
        state = zero_state(2)
        state[0] = 0;state[1] = 1  # |01>
        # Apply SWAP
        swapped_state = apply_gate(state, SWAP, [0, 1], 2)
        # Apply three CNOTs
        cnot_state = apply_gate(state, CNOT, [0, 1], 2)
        cnot_state = apply_gate(cnot_state, NOTC, [0, 1], 2)
        cnot_state = apply_gate(cnot_state, CNOT, [0, 1], 2)
        self.assert_tensors_close(swapped_state, cnot_state)

    def test_reversible_circuit(self):
        state = random_state(2)
        # Apply H on 0, CNOT 0->1
        after = apply_gate(state, H, [0], 2)
        after = apply_gate(after, CNOT, [0, 1], 2)
        # Reverse: CNOT 0->1, H on 0
        reversed = apply_gate(after, CNOT, [0, 1], 2)
        reversed = apply_gate(reversed, H, [0], 2)
        self.assert_tensors_close(state, reversed)

    def test_tensor_logic_3_qubit_x1(self):
        # Test applying gates on different qubits
        state = zero_state(3)
        state = apply_gate(state, X, [0], 3) # |100>
        # Apply X first qubit 000 001 010 011 100
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[4] = 1  # |100>
        self.assert_tensors_close(state, expected)

    def test_tensor_logic_3_qubit_x3(self):
        # Test applying X on qubit 2 (highest index)
        state = zero_state(3)
        # Apply X on qubit 3 (index 0,1,2)
        state = apply_gate(state, X, [2], 3)
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[1] = 1  # |001>
        self.assert_tensors_close(state, expected)
        
    def test_apply_X2(self):
        # Illustrate applying X gate to qubit 2 in a 3-qubit |000> state
        state = zero_state(3)  # |000>
        new_state = apply_gate(state, X, [1], 3)  # Flips qubit 2: |010>
        expected = torch.zeros(8, dtype=torch.cfloat)
        expected[2] = 1  # |010>
        self.assert_tensors_close(new_state, expected)
        
if __name__ == '__main__':
    unittest.main()
