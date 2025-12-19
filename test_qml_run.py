import unittest
import torch
from qml_training import (
    make_real_dataset,
    deep_vqc_forward,
    noisy_qnn_forward,
    quantum_feature_map,
    quantum_kernel,
    kernel_predict,
    EncodingType,
)


class TestSequentialQMLRun(unittest.TestCase):
    def setUp(self):
        # Use real dataset but keep computations tiny by slicing
        X_train, X_test, y_train, y_test = make_real_dataset(test_size=0.9)
        self.X_train = X_train[:5]
        self.X_test = X_test[:5]
        self.y_train = y_train[:5]
        self.y_test = y_test[:5]

    def test_deep_vqc_forward_runs(self):
        x = self.X_train[0]
        # 3 parameters per layer * depth=3 → 9
        theta = torch.randn(9) * 0.1
        out = deep_vqc_forward(x, theta)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.numel(), 1)

    def test_noisy_qnn_forward_runs(self):
        x = self.X_train[0]
        # One layer with 3 params
        theta = torch.randn(3) * 0.1
        out = noisy_qnn_forward(x, theta)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.numel(), 1)

    def test_kernel_functions_run(self):
        x1 = self.X_test[0]
        x2 = self.X_test[1]
        # Feature map + kernel
        psi1 = quantum_feature_map(x1, encoding=EncodingType.ANGLE)
        psi2 = quantum_feature_map(x2, encoding=EncodingType.ANGLE)
        self.assertTrue(torch.is_tensor(psi1))
        self.assertTrue(torch.is_tensor(psi2))
        k = quantum_kernel(x1, x2, encoding=EncodingType.ANGLE)
        self.assertTrue(torch.is_tensor(k))
        self.assertEqual(k.numel(), 1)

    def test_kernel_predict_runs(self):
        x = self.X_test[0]
        pred = kernel_predict(x, self.X_train, self.y_train, encoding=EncodingType.ANGLE)
        self.assertTrue(torch.is_tensor(pred))
        self.assertEqual(pred.numel(), 1)


if __name__ == "__main__":
    unittest.main()
