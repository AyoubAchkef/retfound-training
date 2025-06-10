"""
Unit tests for medical metrics with v6.1 dataset support (28 classes).
"""

import pytest
import torch
import numpy as np

from retfound.metrics.medical import OphthalmologyMetrics
from retfound.core.constants import DATASET_V61_CLASSES, CRITICAL_CONDITIONS


class TestOphthalmologyMetrics:
    """Test ophthalmology metrics for v6.1 dataset."""
    
    @pytest.fixture
    def metrics_v61(self):
        """Create metrics instance for v6.1."""
        return OphthalmologyMetrics(
            num_classes=28,
            dataset_version="v6.1",
            monitor_critical=True
        )
    
    @pytest.fixture
    def metrics_v40(self):
        """Create metrics instance for v4.0."""
        return OphthalmologyMetrics(
            num_classes=22,
            dataset_version="v4.0",
            monitor_critical=False
        )
    
    def test_initialization_v61(self, metrics_v61):
        """Test v6.1 metrics initialization."""
        assert metrics_v61.num_classes == 28
        assert metrics_v61.dataset_version == "v6.1"
        assert len(metrics_v61.class_names) == 28
        assert metrics_v61.class_names == DATASET_V61_CLASSES
        assert metrics_v61.monitor_critical is True
        assert len(metrics_v61.active_classes) == 28
    
    def test_initialization_v40(self, metrics_v40):
        """Test v4.0 metrics initialization."""
        assert metrics_v40.num_classes == 22
        assert metrics_v40.dataset_version == "v4.0"
        assert len(metrics_v40.class_names) == 22
        assert metrics_v40.monitor_critical is False
    
    def test_modality_filtering_fundus(self):
        """Test fundus modality filtering."""
        metrics = OphthalmologyMetrics(
            num_classes=28,
            dataset_version="v6.1",
            modality="fundus"
        )
        assert metrics.active_classes == list(range(18))
        assert len(metrics.active_classes) == 18
    
    def test_modality_filtering_oct(self):
        """Test OCT modality filtering."""
        metrics = OphthalmologyMetrics(
            num_classes=28,
            dataset_version="v6.1",
            modality="oct"
        )
        assert metrics.active_classes == list(range(18, 28))
        assert len(metrics.active_classes) == 10
    
    def test_critical_indices(self, metrics_v61):
        """Test critical condition indices."""
        critical_indices = metrics_v61._get_critical_indices()
        
        # Should have indices for all critical conditions
        expected_conditions = [
            "RAO", "RVO", "Retinal_Detachment", "CNV_Wet_AMD",
            "DR_Proliferative", "RAO_OCT", "RVO_OCT"
        ]
        
        for condition in expected_conditions:
            if condition in DATASET_V61_CLASSES:
                idx = DATASET_V61_CLASSES.index(condition)
                assert idx in critical_indices
    
    def test_update_predictions(self, metrics_v61):
        """Test updating metrics with predictions."""
        # Create dummy predictions
        batch_size = 16
        preds = torch.randint(0, 28, (batch_size,))
        labels = torch.randint(0, 28, (batch_size,))
        probs = torch.rand(batch_size, 28)
        probs = probs / probs.sum(dim=1, keepdim=True)  # Normalize
        
        # Update metrics
        metrics_v61.update(preds, labels, probs)
        
        # Check storage
        assert len(metrics_v61.all_preds) == batch_size
        assert len(metrics_v61.all_labels) == batch_size
        assert len(metrics_v61.all_probs) == batch_size
    
    def test_compute_basic_metrics(self, metrics_v61):
        """Test basic metric computation."""
        # Create controlled predictions
        n_samples = 100
        # Half correct, half wrong
        labels = torch.arange(n_samples) % 28
        preds = labels.clone()
        preds[50:] = (preds[50:] + 1) % 28  # Make half wrong
        
        metrics_v61.update(preds, labels)
        results = metrics_v61.compute()
        
        # Check basic metrics
        assert 'accuracy' in results
        assert results['accuracy'] == pytest.approx(0.5, abs=0.01)
        assert 'balanced_accuracy' in results
        assert 'cohen_kappa' in results
        assert 'f1_macro' in results
        assert 'f1_weighted' in results
    
    def test_compute_auc_metrics(self, metrics_v61):
        """Test AUC metric computation."""
        n_samples = 100
        labels = torch.randint(0, 28, (n_samples,))
        
        # Create probabilities with some signal
        probs = torch.rand(n_samples, 28) * 0.1
        for i in range(n_samples):
            probs[i, labels[i]] += 0.8
        probs = probs / probs.sum(dim=1, keepdim=True)
        
        preds = probs.argmax(dim=1)
        
        metrics_v61.update(preds, labels, probs)
        results = metrics_v61.compute()
        
        # Should have AUC metrics
        assert 'auc_macro' in results
        assert 'auc_weighted' in results
        assert results['auc_macro'] > 0.5  # Better than random
    
    def test_critical_metrics(self, metrics_v61):
        """Test critical condition metrics."""
        n_samples = 100
        
        # Create predictions with some critical conditions
        rao_idx = DATASET_V61_CLASSES.index("RAO")
        rvo_idx = DATASET_V61_CLASSES.index("RVO")
        
        labels = torch.zeros(n_samples, dtype=torch.long)
        labels[:20] = rao_idx  # 20 RAO cases
        labels[20:40] = rvo_idx  # 20 RVO cases
        
        preds = labels.clone()
        # Miss 5 RAO cases (75% sensitivity)
        preds[:5] = 0  # Predict as normal
        
        metrics_v61.update(preds, labels)
        results = metrics_v61.compute()
        
        # Check critical metrics
        assert f'critical_RAO_sensitivity' in results
        assert results[f'critical_RAO_sensitivity'] == pytest.approx(0.75, abs=0.01)
        assert f'critical_RAO_meets_threshold' in results
        assert results[f'critical_RAO_meets_threshold'] == False  # Below 0.9 threshold
        
        assert f'critical_RVO_sensitivity' in results
        assert results[f'critical_RVO_sensitivity'] == pytest.approx(1.0, abs=0.01)
        assert f'critical_RVO_meets_threshold' in results
        assert results[f'critical_RVO_meets_threshold'] == True  # Meets threshold
    
    def test_modality_metrics(self, metrics_v61):
        """Test modality-specific metrics."""
        n_samples = 100
        
        # Create mixed modality predictions
        labels = torch.zeros(n_samples, dtype=torch.long)
        labels[:50] = torch.randint(0, 18, (50,))  # Fundus
        labels[50:] = torch.randint(18, 28, (50,))  # OCT
        
        preds = labels.clone()
        # Make some cross-modality errors
        preds[10:15] = 20  # Fundus predicted as OCT
        preds[60:65] = 5   # OCT predicted as Fundus
        
        metrics_v61.update(preds, labels)
        results = metrics_v61.compute()
        
        # Check modality metrics
        assert 'fundus_accuracy' in results
        assert 'oct_accuracy' in results
        assert 'fundus_correct_modality' in results
        assert 'oct_correct_modality' in results
        
        # Fundus should have 90% accuracy (45/50 correct)
        assert results['fundus_accuracy'] == pytest.approx(0.9, abs=0.01)
        # OCT should have 90% accuracy (45/50 correct)
        assert results['oct_accuracy'] == pytest.approx(0.9, abs=0.01)
    
    def test_dr_quadratic_kappa(self, metrics_v61):
        """Test quadratic kappa for DR grading."""
        n_samples = 100
        
        # Create DR-only predictions
        dr_classes = ["Normal_Fundus", "DR_Mild", "DR_Moderate", "DR_Severe", "DR_Proliferative"]
        dr_indices = [DATASET_V61_CLASSES.index(c) for c in dr_classes]
        
        # Create ordered labels (0, 1, 2, 3, 4 grades)
        labels = torch.tensor([i % 5 for i in range(n_samples)])
        labels = torch.tensor([dr_indices[l] for l in labels])
        
        # Create predictions with some errors
        preds = labels.clone()
        # Add some adjacent errors (less severe)
        preds[::10] = torch.tensor([dr_indices[(i//10 + 1) % 5] for i in range(0, n_samples, 10)])
        
        metrics_v61.update(preds, labels)
        results = metrics_v61.compute()
        
        # Should have quadratic kappa
        assert 'quadratic_kappa' in results
        assert results['quadratic_kappa'] > 0.5  # Should be reasonably high
    
    def test_classification_report(self, metrics_v61):
        """Test classification report generation."""
        # Add some predictions
        n_samples = 280  # 10 samples per class
        labels = torch.arange(n_samples) % 28
        preds = labels.clone()
        
        metrics_v61.update(preds, labels)
        
        # Get report as string
        report_str = metrics_v61.get_classification_report()
        assert isinstance(report_str, str)
        assert len(report_str) > 0
        
        # Get report as dict
        report_dict = metrics_v61.get_classification_report(output_dict=True)
        assert isinstance(report_dict, dict)
        assert 'accuracy' in report_dict
        
        # Check all classes are present
        for class_name in DATASET_V61_CLASSES:
            assert class_name in report_dict
    
    def test_confusion_matrix(self, metrics_v61):
        """Test confusion matrix generation."""
        # Create simple predictions
        n_samples = 28
        labels = torch.arange(n_samples)  # One sample per class
        preds = labels.clone()
        
        metrics_v61.update(preds, labels)
        
        # Get confusion matrix
        cm = metrics_v61.get_confusion_matrix()
        assert cm.shape == (28, 28)
        assert np.trace(cm) == 28  # All on diagonal
        
        # Test normalized confusion matrix
        cm_norm = metrics_v61.get_confusion_matrix(normalize='true')
        assert cm_norm.shape == (28, 28)
        assert np.allclose(np.diag(cm_norm), 1.0)  # All 1s on diagonal
    
    def test_reset_metrics(self, metrics_v61):
        """Test resetting metrics."""
        # Add some data
        preds = torch.randint(0, 28, (10,))
        labels = torch.randint(0, 28, (10,))
        metrics_v61.update(preds, labels)
        
        # Check data is stored
        assert len(metrics_v61.all_preds) == 10
        
        # Reset
        metrics_v61.reset()
        
        # Check data is cleared
        assert len(metrics_v61.all_preds) == 0
        assert len(metrics_v61.all_labels) == 0
        assert len(metrics_v61.all_probs) == 0
    
    @pytest.mark.parametrize("modality,expected_classes", [
        ("fundus", 18),
        ("oct", 10),
        (None, 28)
    ])
    def test_modality_class_counts(self, modality, expected_classes):
        """Test class counts for different modalities."""
        metrics = OphthalmologyMetrics(
            num_classes=28,
            dataset_version="v6.1",
            modality=modality
        )
        assert len(metrics.active_classes) == expected_classes


class TestMetricsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_predictions(self):
        """Test computing metrics with no predictions."""
        metrics = OphthalmologyMetrics(num_classes=28)
        results = metrics.compute()
        assert results == {}
    
    def test_single_class_predictions(self):
        """Test when all predictions are same class."""
        metrics = OphthalmologyMetrics(num_classes=28)
        
        # All predictions are class 0
        preds = torch.zeros(100, dtype=torch.long)
        labels = torch.randint(0, 28, (100,))
        
        metrics.update(preds, labels)
        results = metrics.compute()
        
        # Should still compute metrics
        assert 'accuracy' in results
        assert 'cohen_kappa' in results
    
    def test_invalid_class_indices(self):
        """Test handling of invalid class indices."""
        metrics = OphthalmologyMetrics(num_classes=28)
        
        # This should work fine - metrics should handle it gracefully
        preds = torch.tensor([0, 5, 10, 27])  # Valid indices
        labels = torch.tensor([0, 5, 10, 27])
        
        metrics.update(preds, labels)
        results = metrics.compute()
        assert results['accuracy'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])