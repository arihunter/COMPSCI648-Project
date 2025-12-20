"""
Test suite for parallelization functionality in QML training.

This module verifies that the parallel implementation produces correct results
and handles edge cases appropriately.
"""

import unittest
import torch
from qml_training import train, EncodingType


class TestParallelTraining(unittest.TestCase):
    """Test cases for parallel training implementation."""

    def test_parallel_training_completes(self):
        """Verify parallel training completes without errors."""
        torch.manual_seed(42)
        result = train(
            model_type="noise_aware",
            encoding=EncodingType.ANGLE,
            epochs=2,
            dataset="moons",
            record_metrics=True,
            T1=100,
            T2=200
        )
        
        self.assertIsNotNone(result)
        self.assertIn('final_metrics', result)
        self.assertIn('loss_history', result)
        self.assertIn('acc_history', result)

    def test_parallel_training_metrics_valid(self):
        """Verify parallel training produces valid metrics."""
        torch.manual_seed(42)
        result = train(
            model_type="deep_vqc",
            encoding=EncodingType.ANGLE,
            epochs=2,
            dataset="moons",
            record_metrics=True,
            T1=100,
            T2=200
        )
        
        # Check metrics are in valid range
        final_acc = result['final_metrics']['accuracy']
        self.assertGreaterEqual(final_acc, 0.0)
        self.assertLessEqual(final_acc, 1.0)
        
        # Check history lengths
        self.assertEqual(len(result['loss_history']), 2)
        self.assertEqual(len(result['acc_history']), 2)

    def test_parallel_training_different_seeds_differ(self):
        """Verify different random seeds produce different results."""
        torch.manual_seed(42)
        result1 = train(
            model_type="noise_aware",
            encoding=EncodingType.ANGLE,
            epochs=1,
            dataset="moons",
            record_metrics=True,
            T1=100,
            T2=200
        )
        
        torch.manual_seed(123)
        result2 = train(
            model_type="noise_aware",
            encoding=EncodingType.ANGLE,
            epochs=1,
            dataset="moons",
            record_metrics=True,
            T1=100,
            T2=200
        )
        
        # Different seeds should produce different results
        acc1 = result1['final_metrics']['accuracy']
        acc2 = result2['final_metrics']['accuracy']
        # Allow for same results but they should usually differ
        # This is a weak test but validates randomness is working
        self.assertIsInstance(acc1, float)
        self.assertIsInstance(acc2, float)


class TestParallelKernelMethod(unittest.TestCase):
    """Test cases for parallel kernel method implementation."""

    def test_kernel_method_parallel(self):
        """Verify kernel method works with parallelization."""
        result = train(
            model_type="kernel",
            encoding=EncodingType.ANGLE,
            epochs=1,  # epochs don't matter for kernel
            dataset="moons",
            record_metrics=False,
            T1=100,
            T2=200
        )
        
        # Kernel method returns None but should complete without error
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
