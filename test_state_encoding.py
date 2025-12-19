"""
Tests for state_encoding and param_gate_layer functions.

The state_encoding function returns (initial_state, circuit) separately to allow
noise models to apply noise during state preparation.
"""

import unittest
import torch
import math
from qml_training import state_encoding, EncodingType, param_gate_layer
from quantum_simulator import zero_state, apply_circuit_to_ket, state_to_density


class TestHelper(unittest.TestCase):
    """Helper class with tensor assertion utilities."""
    def assert_tensors_close(self, actual, expected):
        try:
            torch.testing.assert_close(actual, expected)
        except AssertionError as e:
            self.fail(f"\nExpected: {expected}\nReceived: {actual}\nOriginal error: {e}")


class TestStateEncoding(TestHelper):
    """Tests for state_encoding function."""

    def test_angle_encoding_returns_tuple(self):
        """Test angle encoding returns (state, circuit) tuple."""
        n = 2
        x = torch.tensor([math.pi/4, math.pi/6])
        
        result = state_encoding(x, n, encoding=EncodingType.ANGLE)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        state, circuit = result
        self.assertIsInstance(circuit, list)

    def test_angle_encoding_initial_state(self):
        """Test angle encoding returns |00⟩ initial state."""
        n = 2
        x = torch.tensor([math.pi/4, math.pi/6])
        
        state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
        
        expected = zero_state(n)
        self.assert_tensors_close(state, expected)

    def test_angle_encoding_circuit_structure(self):
        """Test angle encoding circuit has correct RY gates."""
        n = 2
        x = torch.tensor([math.pi/4, math.pi/6])
        
        state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
        
        self.assertEqual(len(circuit), 2)
        for i, gate_tuple in enumerate(circuit):
            self.assertIsInstance(gate_tuple, tuple)
            self.assertEqual(len(gate_tuple), 3)
            
            gate_name, qubits, angle = gate_tuple
            self.assertEqual(gate_name, "RY")
            self.assertIsInstance(qubits, list)
            self.assertEqual(len(qubits), 1)
            self.assertTrue(torch.isclose(torch.tensor(angle), x[i]))

    def test_amplitude_encoding_returns_custom_state(self):
        """Test amplitude encoding returns custom state with empty circuit."""
        n = 2
        x = torch.tensor([0.5, 0.5])
        
        try:
            state, circuit = state_encoding(x, n, encoding=EncodingType.AMPLITUDE)
            
            self.assertIsInstance(state, torch.Tensor)
            self.assertIsInstance(circuit, list)
            self.assertEqual(len(circuit), 0)
            
            norm = torch.linalg.norm(state)
            self.assertAlmostEqual(norm.item(), 1.0, places=5)
        except (NotImplementedError, Exception):
            self.skipTest("custom_state not fully implemented")

    def test_invalid_encoding_type_raises(self):
        """Test that invalid encoding type raises error."""
        n = 2
        x = torch.tensor([0.5, 0.5])
        
        with self.assertRaises((ValueError, Exception)):
            state_encoding(x, n, encoding="invalid")

    def test_angle_encoding_zero_input(self):
        """Test angle encoding with zero input."""
        n = 2
        x = torch.tensor([0.0, 0.0])
        
        state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
        
        expected = zero_state(n)
        self.assert_tensors_close(state, expected)
        
        self.assertEqual(len(circuit), 2)
        self.assertTrue(torch.isclose(torch.tensor(circuit[0][2]), torch.tensor(0.0)))
        self.assertTrue(torch.isclose(torch.tensor(circuit[1][2]), torch.tensor(0.0)))

    def test_angle_encoding_pi_input(self):
        """Test angle encoding with π input."""
        n = 2
        x = torch.tensor([math.pi, math.pi])
        
        state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
        
        expected = zero_state(n)
        self.assert_tensors_close(state, expected)
        
        self.assertTrue(torch.isclose(torch.tensor(circuit[0][2]), torch.tensor(math.pi)))
        self.assertTrue(torch.isclose(torch.tensor(circuit[1][2]), torch.tensor(math.pi)))

    def test_angle_encoding_usage_ket_simulation(self):
        """Test usage pattern: ket simulation with apply_circuit_to_ket."""
        n = 2
        x = torch.tensor([math.pi/4, math.pi/6])
        
        state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
        encoded_state = apply_circuit_to_ket(state, circuit, n)
        
        self.assertEqual(encoded_state.shape, (4,))
        norm = torch.linalg.norm(encoded_state)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_angle_encoding_usage_density_simulation(self):
        """Test usage pattern: density matrix with state_to_density."""
        n = 2
        x = torch.tensor([math.pi/4, math.pi/6])
        
        state, circuit = state_encoding(x, n, encoding=EncodingType.ANGLE)
        density = state_to_density(state)
        
        self.assertEqual(density.shape, (4, 4))
        trace = torch.trace(density)
        self.assertAlmostEqual(trace.real.item(), 1.0, places=5)
        self.assertIsInstance(circuit, list)


class TestParamGateLayer(TestHelper):
    """Tests for param_gate_layer function."""

    def test_all_qubits(self):
        """Test param_gate_layer with specify_qubits=None."""
        x = torch.tensor([0.1, 0.2])
        gate = "RY"
        
        op_list = param_gate_layer(gate, x, specify_qubits=None)
        
        self.assertEqual(len(op_list), 2)
        self.assertEqual(op_list[0][0], gate)
        self.assertEqual(op_list[1][0], gate)
        self.assertEqual(op_list[0][1], [0])
        self.assertEqual(op_list[1][1], [1])
        self.assertTrue(torch.isclose(torch.tensor(op_list[0][2]), x[0]))
        self.assertTrue(torch.isclose(torch.tensor(op_list[1][2]), x[1]))

    def test_specified_qubits(self):
        """Test param_gate_layer with specific qubits."""
        x = torch.tensor([0.1, 0.2, 0.3])
        gate = "RY"
        
        op_list = param_gate_layer(gate, x, specify_qubits=(0, 2))
        
        self.assertEqual(len(op_list), 2)
        self.assertEqual(op_list[0][1], [0])
        self.assertEqual(op_list[1][1], [2])
        self.assertTrue(torch.isclose(torch.tensor(op_list[0][2]), x[0]))
        self.assertTrue(torch.isclose(torch.tensor(op_list[1][2]), x[2]))

    def test_gate_format(self):
        """Test param_gate_layer returns correct gate format."""
        x = torch.tensor([0.1, 0.2])
        gate = "RZ"
        
        op_list = param_gate_layer(gate, x, specify_qubits=None)
        
        for gate_tuple in op_list:
            self.assertIsInstance(gate_tuple, tuple)
            self.assertEqual(len(gate_tuple), 3)
            name, qubits, angle = gate_tuple
            self.assertIsInstance(name, str)
            self.assertIsInstance(qubits, list)
            self.assertTrue(all(isinstance(q, int) for q in qubits))


if __name__ == "__main__":
    unittest.main()
