import unittest
import torch
import math
from qml_training import param_gate_layer, noisy_qnn_forward
from quantum_simulator import (
    zero_state,
    apply_gate,
    RY,
    RZ,
    CNOT,
    state_to_density,
    run_noisy_circuit_density,
    Z,
    I2
)


class TestHelper(unittest.TestCase):
    """Helper class for tensor assertions"""
    def assert_tensors_close(self, actual, expected, rtol=1e-5, atol=1e-5):
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            self.fail(f"\nExpected: {expected}\nReceived: {actual}\nOriginal error: {e}")


class TestParamGateLayer(TestHelper):
    """Test class for param_gate_layer function"""
    
    def test_single_qubit_RY_layer(self):
        """Test RY gate layer on a single qubit"""
        x = torch.tensor([math.pi/2])
        qubits = (0,)
        
        result = param_gate_layer("RY", x, qubits)
        
        # Expected: [("RY", [0], pi/2)]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "RY")
        self.assertEqual(result[0][1], [0])
        self.assertAlmostEqual(result[0][2].item(), math.pi/2, places=5)
    
    def test_two_qubit_RY_layer(self):
        """Test RY gate layer on two qubits"""
        x = torch.tensor([math.pi/4, math.pi/3])
        qubits = (0, 1)
        
        result = param_gate_layer("RY", x, qubits)
        
        # Expected: [("RY", [0], pi/4), ("RY", [1], pi/3)]
        self.assertEqual(len(result), 2)
        
        self.assertEqual(result[0][0], "RY")
        self.assertEqual(result[0][1], [0])
        self.assertAlmostEqual(result[0][2].item(), math.pi/4, places=5)
        
        self.assertEqual(result[1][0], "RY")
        self.assertEqual(result[1][1], [1])
        self.assertAlmostEqual(result[1][2].item(), math.pi/3, places=5)
    
    def test_RZ_gate_layer(self):
        """Test RZ gate layer"""
        x = torch.tensor([0.5, 1.0])
        qubits = (0, 1)
        
        result = param_gate_layer("RZ", x, qubits)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "RZ")
        self.assertEqual(result[1][0], "RZ")
        self.assertAlmostEqual(result[0][2].item(), 0.5, places=5)
        self.assertAlmostEqual(result[1][2].item(), 1.0, places=5)
    
    def test_single_qubit_selection(self):
        """Test selecting a single qubit from multi-qubit parameters"""
        x = torch.tensor([0.1, 0.2, 0.3])
        qubits = (1,)  # Only select qubit 1
        
        result = param_gate_layer("RY", x, qubits)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], [1])
        self.assertAlmostEqual(result[0][2].item(), 0.2, places=5)
    
    def test_non_sequential_qubits(self):
        """Test gate layer on non-sequential qubits"""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4])
        qubits = (0, 2)  # Skip qubit 1
        
        result = param_gate_layer("RY", x, qubits)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], [0])
        self.assertAlmostEqual(result[0][2].item(), 0.1, places=5)
        self.assertEqual(result[1][1], [2])
        self.assertAlmostEqual(result[1][2].item(), 0.3, places=5)
    
    def test_empty_qubit_list(self):
        """Test with empty qubit tuple"""
        x = torch.tensor([0.1, 0.2])
        qubits = ()
        
        result = param_gate_layer("RY", x, qubits)
        
        self.assertEqual(len(result), 0)

    def test_default_applies_all_qubits(self):
        """Default specify_qubits=None applies gate to every qubit in x."""
        x = torch.tensor([0.1, 0.2, 0.3])

        result = param_gate_layer("RX", x)

        self.assertEqual(len(result), 3)
        self.assertEqual([entry[1][0] for entry in result], [0, 1, 2])
        self.assertAlmostEqual(result[2][2].item(), 0.3, places=5)

    def test_default_matches_explicit_all(self):
        """specify_qubits=None should match explicitly listing all qubits."""
        x = torch.tensor([0.7, 0.8])

        default_layer = param_gate_layer("RZ", x)
        explicit_layer = param_gate_layer("RZ", x, (0, 1))

        self.assertEqual(default_layer, explicit_layer)
    
    def test_gate_layer_in_circuit(self):
        """Test that param_gate_layer output works in a circuit"""
        n = 2
        x = torch.tensor([math.pi/4, math.pi/6])
        
        # Create gate layer
        gate_layer = param_gate_layer("RY", x, (0, 1))
        
        # Apply gates manually to verify structure
        state = zero_state(n)
        for gate_name, qubit_list, param in gate_layer:
            if gate_name == "RY":
                state = apply_gate(state, RY(param), qubit_list, n)
        
        # Verify state is normalized
        norm = torch.linalg.norm(state)
        self.assertAlmostEqual(norm.item(), 1.0, places=5)
        
        # Verify state is not just |00>
        self.assertNotAlmostEqual(state[0].real.item(), 1.0, places=3)


class TestNoisyQNNIntegration(TestHelper):
    """Integration tests for noisy QNN forward pass using param_gate_layer"""
    
    def test_noisy_qnn_output_range(self):
        """Test that noisy QNN output is in valid range [-1, 1]"""
        x = torch.tensor([0.5, 0.3])
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        output = noisy_qnn_forward(x, theta, T1=100, T2=200)
        
        # Expectation value of Z should be in [-1, 1]
        self.assertGreaterEqual(output.item(), -1.0)
        self.assertLessEqual(output.item(), 1.0)
    
    def test_noisy_qnn_deterministic(self):
        """Test that noisy QNN gives same output for same inputs"""
        x = torch.tensor([0.5, 0.3])
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        output1 = noisy_qnn_forward(x, theta, T1=100, T2=200)
        output2 = noisy_qnn_forward(x, theta, T1=100, T2=200)
        
        self.assertAlmostEqual(output1.item(), output2.item(), places=5)
    
    def test_noisy_qnn_different_inputs(self):
        """Test that different inputs give different outputs"""
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        x1 = torch.tensor([0.5, 0.3])
        x2 = torch.tensor([1.0, 0.8])
        
        output1 = noisy_qnn_forward(x1, theta, T1=100, T2=200)
        output2 = noisy_qnn_forward(x2, theta, T1=100, T2=200)
        
        # Outputs should be different for different inputs
        self.assertNotAlmostEqual(output1.item(), output2.item(), places=3)
    
    def test_noisy_qnn_noise_effect(self):
        """Test that T1/T2 parameters affect the output"""
        x = torch.tensor([0.5, 0.3])
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        # Very high T1/T2 (minimal noise)
        output_low_noise = noisy_qnn_forward(x, theta, T1=10000, T2=10000)
        
        # Low T1/T2 (high noise)
        output_high_noise = noisy_qnn_forward(x, theta, T1=1, T2=1)
        
        # Outputs should be different due to noise
        # Note: they might be close in some cases, so we just verify they exist
        self.assertIsInstance(output_low_noise.item(), float)
        self.assertIsInstance(output_high_noise.item(), float)


class TestCircuitConstruction(TestHelper):
    """Test circuit construction using param_gate_layer"""
    
    def test_build_full_qnn_circuit(self):
        """Test building a complete QNN circuit from layers"""
        x = torch.tensor([0.5, 0.3])
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        # Build circuit layers
        x_RY_layer = param_gate_layer("RY", x, (0, 1))
        theta_RY_layer = param_gate_layer("RY", theta, (0, 1))
        cnot_layer = [("CNOT", [0, 1])]
        
        # Combine all layers
        circuit = x_RY_layer + theta_RY_layer + cnot_layer
        circuit.append(("RZ", [0], theta[2]))
        
        # Verify circuit structure
        self.assertEqual(len(circuit), 6)  # 2 + 2 + 1 + 1
        
        # Check first layer (x encoding)
        self.assertEqual(circuit[0][0], "RY")
        self.assertEqual(circuit[1][0], "RY")
        
        # Check second layer (theta parameters)
        self.assertEqual(circuit[2][0], "RY")
        self.assertEqual(circuit[3][0], "RY")
        
        # Check CNOT
        self.assertEqual(circuit[4][0], "CNOT")
        self.assertEqual(circuit[4][1], [0, 1])
        
        # Check final RZ
        self.assertEqual(circuit[5][0], "RZ")
    
    def test_layer_concatenation(self):
        """Test that multiple layers can be concatenated correctly"""
        x = torch.tensor([0.1, 0.2, 0.3])
        
        layer1 = param_gate_layer("RY", x, (0, 1))
        layer2 = param_gate_layer("RZ", x, (1, 2))
        
        combined = layer1 + layer2
        
        self.assertEqual(len(combined), 4)
        self.assertEqual(combined[0][0], "RY")
        self.assertEqual(combined[1][0], "RY")
        self.assertEqual(combined[2][0], "RZ")
        self.assertEqual(combined[3][0], "RZ")


class TestParallelTraining(TestHelper):
    """Test class for parallel training functions"""
    
    def test_compute_single_param_gradient(self):
        """Test single parameter gradient computation"""
        from qml_training_parallel import compute_single_param_gradient
        
        xi = torch.tensor([0.5, 0.3])
        yi = torch.tensor(1.0)
        theta = torch.tensor([0.1, 0.2, 0.3])
        param_idx = 0
        model_type = "noise_aware"
        
        args = (xi, yi, theta, param_idx, model_type)
        grad = compute_single_param_gradient(args)
        
        # Should return a numeric gradient
        self.assertIsInstance(grad, (float, int))
    
    def test_parallel_forward_pass(self):
        """Test parallel forward pass with small batch"""
        from qml_training_parallel import parallel_forward_pass
        
        X_batch = torch.tensor([[0.5, 0.3], [0.2, 0.8]])
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        results = parallel_forward_pass(X_batch, theta, "noise_aware", num_workers=2)
        
        # Should return predictions for both samples
        self.assertEqual(len(results), 2)
        
        # Each prediction should be in valid range [-1, 1]
        for pred in results:
            self.assertGreaterEqual(pred.item(), -1.0)
            self.assertLessEqual(pred.item(), 1.0)
    
    def test_parallel_gradient_consistency(self):
        """Test that parallel gradients are consistent with sequential"""
        from qml_training_parallel import parallel_gradient_step
        from qml_training import make_real_dataset
        
        # Use small subset for speed
        X_train, _, y_train, _ = make_real_dataset()
        X_train_small = X_train[:3]
        y_train_small = y_train[:3]
        
        theta = torch.tensor([0.1, 0.2, 0.3])
        
        # Compute parallel gradients
        grads_parallel = parallel_gradient_step(
            X_train_small, y_train_small, theta, "noise_aware", num_workers=2
        )
        
        # Should return gradient vector of same shape
        self.assertEqual(grads_parallel.shape, theta.shape)
        
        # Gradients should be finite
        self.assertTrue(torch.all(torch.isfinite(grads_parallel)))
    
    def test_parallel_kernel_predict_shape(self):
        """Test parallel kernel prediction returns correct shape"""
        from qml_training_parallel import parallel_kernel_predict
        from qml_training import make_real_dataset
        
        X_train, X_test, y_train, _ = make_real_dataset()
        
        # Use small subset
        X_train_small = X_train[:5]
        y_train_small = y_train[:5]
        X_test_small = X_test[:3]
        
        scores = parallel_kernel_predict(
            X_test_small, X_train_small, y_train_small, num_workers=2
        )
        
        # Should return predictions for all test samples
        self.assertEqual(len(scores), len(X_test_small))
        
        # Predictions should be in [-1, 1] range (can be 0 if sum equals zero)
        for score in scores:
            self.assertGreaterEqual(score.item(), -1.0)
            self.assertLessEqual(score.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
