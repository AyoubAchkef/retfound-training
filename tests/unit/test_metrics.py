"""
Metrics Tests
=============

Test medical imaging metrics and evaluation functions.
"""

import pytest
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score

from retfound.metrics.medical import (
    OphthalmologyMetrics,
    compute_sensitivity_specificity,
    compute_ppv_npv,
    compute_quadratic_kappa,
    compute_auc_ci,
    CalibrationMetrics
)
from retfound.metrics.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    create_metrics_report
)


class TestOphthalmologyMetrics:
    """Test ophthalmology-specific metrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        num_classes = 5
        class_names = ['Normal', 'DR_Mild', 'DR_Moderate', 'DR_Severe', 'DR_Proliferative']
        
        metrics = OphthalmologyMetrics(
            num_classes=num_classes,
            class_names=class_names
        )
        
        assert metrics.num_classes == num_classes
        assert metrics.class_names == class_names
        assert len(metrics.predictions) == 0
        assert len(metrics.targets) == 0
    
    def test_metrics_update(self):
        """Test updating metrics with batch"""
        metrics = OphthalmologyMetrics(num_classes=3)
        
        # Create batch
        batch_size = 4
        outputs = torch.randn(batch_size, 3)
        targets = torch.tensor([0, 1, 2, 0])
        
        # Update
        metrics.update(outputs, targets)
        
        assert len(metrics.predictions) == batch_size
        assert len(metrics.targets) == batch_size
        assert len(metrics.probabilities) == batch_size
    
    def test_compute_basic_metrics(self):
        """Test computing basic metrics"""
        metrics = OphthalmologyMetrics(num_classes=3)
        
        # Add perfect predictions
        outputs = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        targets = torch.tensor([0, 1, 2])
        
        metrics.update(outputs, targets)
        results = metrics.compute_metrics()
        
        # Perfect predictions should give 100% accuracy
        assert results['accuracy'] == 100.0
        assert results['balanced_accuracy'] == 100.0
    
    def test_sensitivity_specificity(self):
        """Test per-class sensitivity and specificity"""
        metrics = OphthalmologyMetrics(
            num_classes=2,
            class_names=['Normal', 'Disease']
        )
        
        # Add predictions
        # True positives and negatives, with some errors
        outputs = torch.tensor([
            [10.0, 0.0],   # Correct negative
            [0.0, 10.0],   # Correct positive
            [10.0, 0.0],   # Correct negative
            [5.0, 4.0],    # False negative
            [4.0, 5.0],    # Correct positive
            [4.0, 5.0],    # False positive
        ])
        targets = torch.tensor([0, 1, 0, 1, 1, 0])
        
        metrics.update(outputs, targets)
        results = metrics.compute_metrics()
        
        # Check Disease class metrics
        disease_sens = results['Disease_sensitivity']
        disease_spec = results['Disease_specificity']
        
        # Sensitivity = TP / (TP + FN) = 2 / (2 + 1) = 66.7%
        assert abs(disease_sens - 66.7) < 1.0
        
        # Specificity = TN / (TN + FP) = 2 / (2 + 1) = 66.7%
        assert abs(disease_spec - 66.7) < 1.0
    
    def test_cohen_kappa(self):
        """Test Cohen's kappa calculation"""
        metrics = OphthalmologyMetrics(num_classes=3)
        
        # Add predictions with some agreement
        predictions = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        targets = [0, 1, 2, 0, 1, 1, 0, 2, 2]  # Some disagreements
        
        for i in range(len(predictions)):
            output = torch.zeros(3)
            output[predictions[i]] = 10.0
            metrics.update(output.unsqueeze(0), torch.tensor([targets[i]]))
        
        results = metrics.compute_metrics()
        
        # Calculate expected kappa
        expected_kappa = cohen_kappa_score(targets, predictions)
        assert abs(results['cohen_kappa'] - expected_kappa) < 0.01
    
    def test_auc_calculation(self):
        """Test AUC-ROC calculation"""
        metrics = OphthalmologyMetrics(num_classes=2)
        
        # Add predictions with varying confidence
        np.random.seed(42)
        n_samples = 100
        
        for i in range(n_samples):
            if i < 50:
                # Class 0 - higher probability for class 0
                logits = torch.tensor([[np.random.randn() + 2, np.random.randn() - 2]])
                target = 0
            else:
                # Class 1 - higher probability for class 1
                logits = torch.tensor([[np.random.randn() - 2, np.random.randn() + 2]])
                target = 1
            
            metrics.update(logits, torch.tensor([target]))
        
        results = metrics.compute_metrics()
        
        # Should have good AUC for this separable data
        assert results['auc_macro'] > 0.8
        assert 'Class_0_auc' in results
        assert 'Class_1_auc' in results
    
    def test_quadratic_kappa(self):
        """Test quadratic kappa for DR grading"""
        metrics = OphthalmologyMetrics(
            num_classes=5,
            class_names=['DR_0', 'DR_1', 'DR_2', 'DR_3', 'DR_4']
        )
        
        # Simulate DR grading predictions
        # Quadratic kappa penalizes based on distance
        predictions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        targets = [0, 1, 2, 3, 4, 1, 2, 3, 4, 3]  # Last few are off by 1
        
        for pred, target in zip(predictions, targets):
            output = torch.zeros(5)
            output[pred] = 10.0
            metrics.update(output.unsqueeze(0), torch.tensor([target]))
        
        results = metrics.compute_metrics()
        
        # Check if DR quadratic kappa is computed
        if 'dr_quadratic_kappa' in results:
            # Off-by-one errors should give reasonable kappa
            assert results['dr_quadratic_kappa'] > 0.7
    
    def test_critical_conditions_alerts(self):
        """Test critical condition performance alerts"""
        metrics = OphthalmologyMetrics(
            num_classes=3,
            class_names=['Normal', 'RAO', 'RVO']  # Critical conditions
        )
        
        # Add predictions with low sensitivity for critical conditions
        # Many false negatives for RAO
        for _ in range(20):
            # Normal correctly classified
            metrics.update(torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([0]))
        
        for _ in range(10):
            # RAO misclassified as normal (false negative)
            metrics.update(torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([1]))
        
        for _ in range(2):
            # RAO correctly classified
            metrics.update(torch.tensor([[0.0, 10.0, 0.0]]), torch.tensor([1]))
        
        results = metrics.compute_metrics()
        
        # Should have critical alerts
        assert 'critical_alerts' in results
        alerts = results['critical_alerts']
        
        # RAO should be flagged for low sensitivity
        rao_alert = next((a for a in alerts if 'RAO' in a['class']), None)
        assert rao_alert is not None
        assert rao_alert['current_sensitivity'] < rao_alert['required_sensitivity']


class TestMedicalMetricFunctions:
    """Test individual medical metric functions"""
    
    def test_sensitivity_specificity_calculation(self):
        """Test sensitivity and specificity calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0])
        
        sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred)
        
        # Sensitivity = TP / (TP + FN) = 3 / 4 = 0.75
        assert abs(sensitivity - 0.75) < 0.01
        
        # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
        assert abs(specificity - 0.75) < 0.01
    
    def test_ppv_npv_calculation(self):
        """Test PPV and NPV calculation"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0])
        
        ppv, npv = compute_ppv_npv(y_true, y_pred)
        
        # PPV = TP / (TP + FP) = 3 / 4 = 0.75
        assert abs(ppv - 0.75) < 0.01
        
        # NPV = TN / (TN + FN) = 3 / 4 = 0.75
        assert abs(npv - 0.75) < 0.01
    
    def test_quadratic_kappa_function(self):
        """Test quadratic kappa function"""
        # Perfect agreement
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        
        kappa_perfect = compute_quadratic_kappa(y_true, y_pred)
        assert kappa_perfect == 1.0
        
        # Off by one errors
        y_true = [0, 1, 2, 3, 4]
        y_pred = [1, 2, 3, 4, 3]
        
        kappa_off_by_one = compute_quadratic_kappa(y_true, y_pred)
        assert 0.5 < kappa_off_by_one < 0.9
        
        # Random predictions
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 0, 3, 1, 2]
        
        kappa_random = compute_quadratic_kappa(y_true, y_pred)
        assert kappa_random < 0.5
    
    def test_auc_confidence_interval(self):
        """Test AUC with confidence interval calculation"""
        np.random.seed(42)
        n_samples = 100
        
        # Generate separable data
        y_true = np.array([0] * 50 + [1] * 50)
        y_scores = np.concatenate([
            np.random.randn(50) - 1,  # Lower scores for class 0
            np.random.randn(50) + 1   # Higher scores for class 1
        ])
        
        auc, ci_lower, ci_upper = compute_auc_ci(
            y_true, y_scores,
            n_bootstraps=100,
            ci_level=0.95
        )
        
        # Check AUC is reasonable
        assert 0.7 < auc < 1.0
        
        # Check CI is reasonable
        assert ci_lower < auc < ci_upper
        assert ci_upper - ci_lower < 0.2  # CI shouldn't be too wide


class TestCalibrationMetrics:
    """Test probability calibration metrics"""
    
    def test_calibration_metrics(self):
        """Test calibration metric computation"""
        cal_metrics = CalibrationMetrics(n_bins=10)
        
        # Perfect calibration
        n_samples = 100
        probabilities = np.linspace(0, 1, n_samples)
        # Create labels that match probabilities
        labels = (probabilities > 0.5).astype(int)
        
        # Add some randomness but maintain calibration
        np.random.seed(42)
        for i in range(n_samples):
            if np.random.rand() < probabilities[i]:
                labels[i] = 1
            else:
                labels[i] = 0
        
        results = cal_metrics.compute(
            y_true=labels,
            y_prob=probabilities
        )
        
        assert 'ece' in results  # Expected Calibration Error
        assert 'mce' in results  # Maximum Calibration Error
        assert 'reliability_diagram' in results
        
        # ECE should be relatively low for calibrated predictions
        assert results['ece'] < 0.2
    
    def test_temperature_scaling(self):
        """Test temperature scaling for calibration"""
        from retfound.metrics.medical import TemperatureScaler
        
        # Overconfident predictions (too peaked)
        logits = torch.tensor([
            [5.0, 0.0],
            [0.0, 5.0],
            [4.0, 1.0],
            [1.0, 4.0]
        ])
        labels = torch.tensor([0, 1, 0, 1])
        
        # Fit temperature
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        
        # Temperature should be > 1 for overconfident predictions
        assert scaler.temperature > 1.0
        
        # Scaled predictions should be less confident
        scaled_logits = scaler.transform(logits)
        
        original_probs = torch.softmax(logits, dim=1)
        scaled_probs = torch.softmax(scaled_logits, dim=1)
        
        # Max probability should decrease after scaling
        assert scaled_probs.max(dim=1)[0].mean() < original_probs.max(dim=1)[0].mean()


class TestMetricsVisualization:
    """Test metrics visualization functions"""
    
    def test_confusion_matrix_plot(self, temp_output_dir):
        """Test confusion matrix plotting"""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
        y_pred = [0, 1, 2, 0, 2, 2, 0, 1, 1, 1]
        class_names = ['Class A', 'Class B', 'Class C']
        
        save_path = temp_output_dir / 'confusion_matrix.png'
        
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=save_path
        )
        
        assert save_path.exists()
    
    def test_roc_curves_plot(self, temp_output_dir):
        """Test ROC curves plotting"""
        # Generate sample data
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        
        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob = np.random.rand(n_samples, n_classes)
        # Normalize to sum to 1
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        save_path = temp_output_dir / 'roc_curves.png'
        
        plot_roc_curves(
            y_true=y_true,
            y_prob=y_prob,
            class_names=['Class A', 'Class B', 'Class C'],
            save_path=save_path
        )
        
        assert save_path.exists()
    
    def test_create_metrics_report(self, temp_output_dir, sample_metrics):
        """Test creating comprehensive metrics report"""
        report_path = temp_output_dir / 'metrics_report.html'
        
        create_metrics_report(
            metrics=sample_metrics,
            save_path=report_path,
            model_name='Test Model',
            dataset_name='Test Dataset'
        )
        
        assert report_path.exists()
        
        # Check report contains key information
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Test Model' in content
            assert 'accuracy' in content
            assert 'sensitivity' in content
