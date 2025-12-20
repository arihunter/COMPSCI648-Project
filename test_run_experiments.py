"""
Unit tests for run_experiments.py

Tests verify:
- Training runs complete successfully
- CSV files are created with correct format
- Metrics are recorded properly
- File naming conventions are correct
"""

import unittest
import os
import csv
import shutil
import tempfile
from datetime import datetime

from qml_training import train, EncodingType
from run_experiments import (
    ensure_data_dir,
    save_training_history_csv,
    save_summary_csv,
    run_all_experiments,
    MODEL_TYPES,
    DATASETS
)


class TestTrainFunction(unittest.TestCase):
    """Test the train function with record_metrics=True."""
    
    def test_train_returns_metrics_dict(self):
        """Verify train returns a dict with expected keys when record_metrics=True."""
        metrics = train(
            model_type="deep_vqc",
            encoding=EncodingType.ANGLE,
            epochs=2,
            dataset="moons",
            record_metrics=True
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss_history', metrics)
        self.assertIn('acc_history', metrics)
        self.assertIn('final_metrics', metrics)
        self.assertIn('model_type', metrics)
        self.assertIn('dataset', metrics)
        self.assertIn('encoding', metrics)
        self.assertIn('epochs', metrics)
    
    def test_train_loss_history_length(self):
        """Verify loss_history has correct number of epochs."""
        epochs = 2
        metrics = train(
            model_type="deep_vqc",
            encoding=EncodingType.ANGLE,
            epochs=epochs,
            dataset="moons",
            record_metrics=True
        )
        
        self.assertEqual(len(metrics['loss_history']), epochs)
        self.assertEqual(len(metrics['acc_history']), epochs)
    
    def test_train_accuracy_in_valid_range(self):
        """Verify accuracy values are between 0 and 1."""
        metrics = train(
            model_type="noise_aware",
            encoding=EncodingType.ANGLE,
            epochs=2,
            dataset="moons",
            record_metrics=True
        )
        
        for acc in metrics['acc_history']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
    
    def test_train_final_metrics_keys(self):
        """Verify final_metrics contains expected metric keys."""
        metrics = train(
            model_type="deep_vqc",
            encoding=EncodingType.ANGLE,
            epochs=2,
            dataset="moons",
            record_metrics=True
        )
        
        final = metrics['final_metrics']
        self.assertIn('accuracy', final)
        self.assertIn('precision', final)
        self.assertIn('recall', final)
        self.assertIn('f1', final)
        self.assertIn('roc_auc', final)


class TestSaveTrainingHistoryCSV(unittest.TestCase):
    """Test the save_training_history_csv function."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_csv_file_created(self):
        """Verify CSV file is created."""
        metrics = {
            'loss_history': [1.0, 0.8, 0.6],
            'acc_history': [0.5, 0.7, 0.85]
        }
        filepath = os.path.join(self.test_dir, "test_history.csv")
        
        save_training_history_csv(metrics, filepath)
        
        self.assertTrue(os.path.exists(filepath))
    
    def test_csv_has_correct_headers(self):
        """Verify CSV has correct column headers."""
        metrics = {
            'loss_history': [1.0, 0.8],
            'acc_history': [0.5, 0.7]
        }
        filepath = os.path.join(self.test_dir, "test_history.csv")
        
        save_training_history_csv(metrics, filepath)
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, ['epoch', 'loss', 'acc'])
    
    def test_csv_has_correct_row_count(self):
        """Verify CSV has correct number of data rows."""
        metrics = {
            'loss_history': [1.0, 0.8, 0.6],
            'acc_history': [0.5, 0.7, 0.85]
        }
        filepath = os.path.join(self.test_dir, "test_history.csv")
        
        save_training_history_csv(metrics, filepath)
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # 1 header + 3 data rows
            self.assertEqual(len(rows), 4)
    
    def test_csv_epoch_numbers_correct(self):
        """Verify epoch numbers start at 1 and increment."""
        metrics = {
            'loss_history': [1.0, 0.8, 0.6],
            'acc_history': [0.5, 0.7, 0.85]
        }
        filepath = os.path.join(self.test_dir, "test_history.csv")
        
        save_training_history_csv(metrics, filepath)
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for i, row in enumerate(reader, start=1):
                self.assertEqual(int(row[0]), i)


class TestSaveSummaryCSV(unittest.TestCase):
    """Test the save_summary_csv function."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_summary_csv_created(self):
        """Verify summary CSV file is created."""
        results = [{
            'model_type': 'deep_vqc',
            'encoding': 'angle',
            'dataset': 'moons',
            'loss_history': [1.0, 0.5],
            'acc_history': [0.6, 0.8],
            'final_metrics': {'f1': 0.75, 'roc_auc': 0.82}
        }]
        filepath = os.path.join(self.test_dir, "summary.csv")
        
        save_summary_csv(results, filepath)
        
        self.assertTrue(os.path.exists(filepath))
    
    def test_summary_csv_headers(self):
        """Verify summary CSV has correct headers."""
        results = [{
            'model_type': 'deep_vqc',
            'encoding': 'angle',
            'dataset': 'moons',
            'loss_history': [0.5],
            'acc_history': [0.8],
            'final_metrics': {'f1': 0.75, 'roc_auc': 0.82}
        }]
        filepath = os.path.join(self.test_dir, "summary.csv")
        
        save_summary_csv(results, filepath)
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, ['model', 'enc', 'data', 'loss', 'acc', 'f1', 'auc'])
    
    def test_summary_csv_multiple_results(self):
        """Verify summary CSV handles multiple results."""
        results = [
            {
                'model_type': 'deep_vqc',
                'encoding': 'angle',
                'dataset': 'moons',
                'loss_history': [0.5],
                'acc_history': [0.8],
                'final_metrics': {'f1': 0.75, 'roc_auc': 0.82}
            },
            {
                'model_type': 'noise_aware',
                'encoding': 'amplitude',
                'dataset': 'real',
                'loss_history': [0.6],
                'acc_history': [0.7],
                'final_metrics': {'f1': 0.65, 'roc_auc': 0.72}
            }
        ]
        filepath = os.path.join(self.test_dir, "summary.csv")
        
        save_summary_csv(results, filepath)
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # 1 header + 2 data rows
            self.assertEqual(len(rows), 3)


class TestEnsureDataDir(unittest.TestCase):
    """Test the ensure_data_dir function."""
    
    def test_data_dir_created(self):
        """Verify data directory is created if it doesn't exist."""
        data_dir = ensure_data_dir()
        self.assertTrue(os.path.isdir(data_dir))
        self.assertTrue(data_dir.endswith("data"))


class TestRunAllExperiments(unittest.TestCase):
    """Integration tests for run_all_experiments with minimal epochs."""
    
    @classmethod
    def setUpClass(cls):
        """Run experiments once for all tests in this class."""
        # Use minimal settings for speed
        import run_experiments
        # Temporarily reduce configurations for testing
        cls.original_model_types = run_experiments.MODEL_TYPES
        cls.original_encodings = run_experiments.ENCODINGS
        cls.original_datasets = run_experiments.DATASETS
        
        # Only test one model, one encoding, one dataset for speed
        run_experiments.MODEL_TYPES = ["deep_vqc"]
        run_experiments.ENCODINGS = [EncodingType.ANGLE]
        run_experiments.DATASETS = ["moons"]
        
        cls.results = run_all_experiments(epochs=2, T1=100, T2=200)
        
    @classmethod
    def tearDownClass(cls):
        """Restore original configurations."""
        import run_experiments
        run_experiments.MODEL_TYPES = cls.original_model_types
        run_experiments.ENCODINGS = cls.original_encodings
        run_experiments.DATASETS = cls.original_datasets
    
    def test_returns_results_list(self):
        """Verify run_all_experiments returns a list."""
        self.assertIsInstance(self.results, list)
    
    def test_results_not_empty(self):
        """Verify at least one result is returned."""
        self.assertGreater(len(self.results), 0)
    
    def test_result_has_required_keys(self):
        """Verify each result has required keys."""
        for result in self.results:
            self.assertIn('loss_history', result)
            self.assertIn('acc_history', result)
            self.assertIn('final_metrics', result)
            self.assertIn('model_type', result)
            self.assertIn('dataset', result)
            self.assertIn('T1', result)
            self.assertIn('T2', result)
    
    def test_csv_files_created(self):
        """Verify CSV files are created in data directory."""
        data_dir = ensure_data_dir()
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0)
    
    def test_history_csv_format(self):
        """Verify history CSV files have correct format."""
        data_dir = ensure_data_dir()
        csv_files = [f for f in os.listdir(data_dir) 
                     if f.endswith('.csv') and not f.startswith('summary')]
        
        if csv_files:
            filepath = os.path.join(data_dir, csv_files[0])
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                self.assertEqual(headers, ['epoch', 'loss', 'acc'])


class TestFileNamingConvention(unittest.TestCase):
    """Test that file naming follows the expected convention."""
    
    def test_history_filename_format(self):
        """Verify history filename contains expected components."""
        data_dir = ensure_data_dir()
        csv_files = [f for f in os.listdir(data_dir) 
                     if f.endswith('.csv') and not f.startswith('summary')]
        
        if csv_files:
            filename = csv_files[0]
            # Should contain: model_dataset_t1{T1}_t2{T2}_ep{epochs}_{date}.csv
            self.assertIn('_t1', filename)
            self.assertIn('_t2', filename)
            self.assertIn('_ep', filename)
    
    def test_summary_filename_format(self):
        """Verify summary filename contains expected components."""
        data_dir = ensure_data_dir()
        summary_files = [f for f in os.listdir(data_dir) 
                         if f.startswith('summary') and f.endswith('.csv')]
        
        if summary_files:
            filename = summary_files[0]
            # Should contain: summary_t1{T1}_t2{T2}_ep{epochs}_{date}.csv
            self.assertIn('summary_t1', filename)
            self.assertIn('_t2', filename)
            self.assertIn('_ep', filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
