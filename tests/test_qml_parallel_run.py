import unittest
import torch
from qml_training_parallel import (
    parallel_forward_pass,
    parallel_gradient_step,
    train_kernel_parallel,
)
from qml_training import make_real_dataset


class TestParallelQMLRun(unittest.TestCase):
    def setUp(self):
        X_train, X_test, y_train, y_test = make_real_dataset(test_size=0.9)
        # Keep tiny slices for fast tests
        self.X_train = X_train[:5]
        self.X_test = X_test[:4]
        self.y_train = y_train[:5]
        self.y_test = y_test[:4]
        self.num_workers = 2

    def test_parallel_forward_pass_deep_vqc_runs(self):
        theta = torch.randn(9) * 0.1
        scores = parallel_forward_pass(self.X_test, theta, model_type="deep_vqc", num_workers=self.num_workers)
        self.assertTrue(torch.is_tensor(scores))
        self.assertEqual(scores.shape[0], len(self.X_test))

    def test_parallel_forward_pass_noise_aware_runs(self):
        theta = torch.randn(3) * 0.1
        scores = parallel_forward_pass(self.X_test, theta, model_type="noise_aware", num_workers=self.num_workers)
        self.assertTrue(torch.is_tensor(scores))
        self.assertEqual(scores.shape[0], len(self.X_test))

    def test_parallel_gradient_step_deep_vqc_runs(self):
        theta = torch.randn(9) * 0.1
        grads = parallel_gradient_step(self.X_train, self.y_train, theta, model_type="deep_vqc", num_workers=self.num_workers)
        self.assertTrue(torch.is_tensor(grads))
        self.assertEqual(grads.shape, theta.shape)

    def test_parallel_gradient_step_noise_aware_runs(self):
        theta = torch.randn(3) * 0.1
        grads = parallel_gradient_step(self.X_train, self.y_train, theta, model_type="noise_aware", num_workers=self.num_workers)
        self.assertTrue(torch.is_tensor(grads))
        self.assertEqual(grads.shape, theta.shape)

    def test_kernel_parallel_metrics_runs(self):
        # This uses parallel prediction; it prints metrics and returns a dict
        metrics = train_kernel_parallel(num_workers=self.num_workers)
        self.assertIsInstance(metrics, dict)
        # Expect common keys
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
