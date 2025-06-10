"""
Loss Function Tests
===================

Test custom loss functions for medical imaging.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from retfound.training.losses import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    AsymmetricLoss,
    DiceLoss,
    CombinedLoss,
    create_loss_function,
    MixupCriterion,
    CutMixCriterion
)
from retfound.core.config import RETFoundConfig


class TestLabelSmoothing:
    """Test label smoothing cross entropy"""
    
    def test_label_smoothing_creation(self):
        """Test creating label smoothing loss"""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        assert hasattr(loss_fn, 'smoothing')
        assert loss_fn.smoothing == 0.1
        assert loss_fn.confidence == 0.9
    
    def test_label_smoothing_forward(self):
        """Test label smoothing forward pass"""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Create dummy predictions and targets
        batch_size = 4
        num_classes = 10
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Compute loss
        loss = loss_fn(predictions, targets)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss > 0
        assert not torch.isnan(loss)
    
    def test_label_smoothing_vs_regular_ce(self):
        """Test label smoothing reduces confidence"""
        batch_size = 4
        num_classes = 10
        
        # Create peaked predictions (high confidence)
        predictions = torch.zeros(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        for i in range(batch_size):
            predictions[i, targets[i]] = 10.0  # High logit for correct class
        
        # Regular CE
        ce_loss = F.cross_entropy(predictions, targets)
        
        # Label smoothing CE
        ls_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        ls_loss = ls_loss_fn(predictions, targets)
        
        # Label smoothing should give higher loss (penalizes overconfidence)
        assert ls_loss > ce_loss


class TestFocalLoss:
    """Test focal loss for class imbalance"""
    
    def test_focal_loss_creation(self):
        """Test creating focal loss"""
        loss_fn = FocalLoss(alpha=None, gamma=2.0)
        
        assert loss_fn.gamma == 2.0
        assert loss_fn.alpha is None
        
        # With class weights
        alpha = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        assert torch.equal(loss_fn.alpha, alpha)
    
    def test_focal_loss_forward(self):
        """Test focal loss forward pass"""
        num_classes = 3
        loss_fn = FocalLoss(gamma=2.0)
        
        # Easy case - high confidence correct predictions
        predictions_easy = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets_easy = torch.tensor([0, 1])
        
        # Hard case - low confidence correct predictions
        predictions_hard = torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        targets_hard = torch.tensor([0, 1])
        
        loss_easy = loss_fn(predictions_easy, targets_easy)
        loss_hard = loss_fn(predictions_hard, targets_hard)
        
        # Focal loss should penalize hard examples more
        assert loss_hard > loss_easy
    
    def test_focal_loss_with_alpha(self):
        """Test focal loss with class weights"""
        # Imbalanced scenario
        alpha = torch.tensor([0.25, 0.75])  # Class 0 is majority, class 1 is minority
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        
        predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # Loss for majority class
        loss_majority = loss_fn(predictions[0:1], torch.tensor([0]))
        
        # Loss for minority class  
        loss_minority = loss_fn(predictions[1:2], torch.tensor([1]))
        
        # Minority class should have higher weight
        assert loss_minority > loss_majority


class TestAsymmetricLoss:
    """Test asymmetric loss for multi-label classification"""
    
    def test_asymmetric_loss_creation(self):
        """Test creating asymmetric loss"""
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        
        assert loss_fn.gamma_neg == 4
        assert loss_fn.gamma_pos == 1
        assert loss_fn.clip == 0.05
    
    def test_asymmetric_loss_forward(self):
        """Test asymmetric loss forward pass"""
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        
        # Multi-label predictions
        batch_size = 4
        num_classes = 5
        predictions = torch.randn(batch_size, num_classes)
        
        # Multi-label targets (0 or 1 for each class)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss > 0
        assert not torch.isnan(loss)
    
    def test_asymmetric_loss_properties(self):
        """Test asymmetric loss focuses on hard negatives"""
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        
        predictions = torch.tensor([[5.0, -5.0, 0.0]])  # High pos, low neg, neutral
        
        # Test with positive label
        targets_pos = torch.tensor([[1.0, 0.0, 0.0]])
        loss_pos = loss_fn(predictions, targets_pos)
        
        # Test with negative label  
        targets_neg = torch.tensor([[0.0, 1.0, 0.0]])
        loss_neg = loss_fn(predictions, targets_neg)
        
        # Both should be valid losses
        assert loss_pos > 0
        assert loss_neg > 0


class TestDiceLoss:
    """Test Dice loss for segmentation"""
    
    def test_dice_loss_creation(self):
        """Test creating Dice loss"""
        loss_fn = DiceLoss(smooth=1.0)
        
        assert loss_fn.smooth == 1.0
    
    def test_dice_loss_binary(self):
        """Test binary Dice loss"""
        loss_fn = DiceLoss(smooth=1.0)
        
        # Perfect prediction
        predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        
        loss = loss_fn(predictions, targets)
        
        # Should be close to 0 for perfect prediction
        assert loss < 0.1
        
        # Bad prediction
        predictions_bad = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss_bad = loss_fn(predictions_bad, targets)
        
        # Should be close to 1 for completely wrong prediction
        assert loss_bad > 0.9
    
    def test_dice_loss_multiclass(self):
        """Test multi-class Dice loss"""
        loss_fn = DiceLoss(smooth=1.0)
        
        batch_size = 4
        num_classes = 3
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss <= 1  # Dice loss is bounded


class TestCombinedLoss:
    """Test combined loss functions"""
    
    def test_combined_loss_creation(self):
        """Test creating combined loss"""
        loss1 = nn.CrossEntropyLoss()
        loss2 = FocalLoss(gamma=2.0)
        
        combined = CombinedLoss(
            losses=[loss1, loss2],
            weights=[0.7, 0.3]
        )
        
        assert len(combined.losses) == 2
        assert combined.weights == [0.7, 0.3]
    
    def test_combined_loss_forward(self):
        """Test combined loss forward pass"""
        loss1 = nn.CrossEntropyLoss()
        loss2 = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        combined = CombinedLoss(
            losses=[loss1, loss2],
            weights=[0.5, 0.5]
        )
        
        predictions = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        
        # Individual losses
        l1 = loss1(predictions, targets)
        l2 = loss2(predictions, targets)
        
        # Combined loss
        loss = combined(predictions, targets)
        
        # Should be weighted average
        expected = 0.5 * l1 + 0.5 * l2
        assert torch.allclose(loss, expected)


class TestMixupCriterion:
    """Test Mixup loss criterion"""
    
    def test_mixup_criterion(self):
        """Test Mixup criterion"""
        base_criterion = nn.CrossEntropyLoss()
        criterion = MixupCriterion(base_criterion)
        
        batch_size = 4
        num_classes = 10
        predictions = torch.randn(batch_size, num_classes)
        
        # Mixup targets
        targets_a = torch.randint(0, num_classes, (batch_size,))
        targets_b = torch.randint(0, num_classes, (batch_size,))
        lam = 0.7
        
        loss = criterion(predictions, targets_a, targets_b, lam)
        
        # Check it's a weighted combination
        loss_a = base_criterion(predictions, targets_a)
        loss_b = base_criterion(predictions, targets_b)
        expected = lam * loss_a + (1 - lam) * loss_b
        
        assert torch.allclose(loss, expected)


class TestCutMixCriterion:
    """Test CutMix loss criterion"""
    
    def test_cutmix_criterion(self):
        """Test CutMix criterion"""
        base_criterion = nn.CrossEntropyLoss()
        criterion = CutMixCriterion(base_criterion)
        
        batch_size = 4
        num_classes = 10
        predictions = torch.randn(batch_size, num_classes)
        
        # CutMix uses same format as Mixup
        targets_a = torch.randint(0, num_classes, (batch_size,))
        targets_b = torch.randint(0, num_classes, (batch_size,))
        lam = 0.6
        
        loss = criterion(predictions, targets_a, targets_b, lam)
        
        assert isinstance(loss, torch.Tensor)
        assert loss > 0


class TestLossFactory:
    """Test loss function factory"""
    
    def test_create_standard_losses(self, minimal_config):
        """Test creating standard loss functions"""
        # Cross entropy
        minimal_config.label_smoothing = 0.0
        minimal_config.use_focal_loss = False
        loss_fn = create_loss_function(minimal_config)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        
        # Label smoothing
        minimal_config.label_smoothing = 0.1
        loss_fn = create_loss_function(minimal_config)
        assert isinstance(loss_fn, LabelSmoothingCrossEntropy)
        
        # Focal loss
        minimal_config.label_smoothing = 0.0
        minimal_config.use_focal_loss = True
        loss_fn = create_loss_function(minimal_config)
        assert isinstance(loss_fn, FocalLoss)
    
    def test_create_weighted_loss(self, minimal_config):
        """Test creating weighted loss"""
        minimal_config.use_class_weights = True
        minimal_config.class_weights = torch.tensor([1.0, 2.0, 3.0])
        
        loss_fn = create_loss_function(minimal_config)
        
        if hasattr(loss_fn, 'weight'):
            assert torch.equal(loss_fn.weight, minimal_config.class_weights)


class TestLossUtils:
    """Test loss utility functions"""
    
    def test_compute_class_weights(self):
        """Test computing class weights from distribution"""
        from retfound.training.losses import compute_class_weights
        
        # Imbalanced class distribution
        class_counts = [1000, 500, 100, 50]
        
        weights = compute_class_weights(class_counts, method='inverse')
        
        # Rare classes should have higher weights
        assert weights[3] > weights[2] > weights[1] > weights[0]
        
        # Test effective number method
        weights_eff = compute_class_weights(class_counts, method='effective', beta=0.99)
        assert all(w > 0 for w in weights_eff)
    
    def test_reduce_loss(self):
        """Test loss reduction methods"""
        from retfound.training.losses import reduce_loss
        
        # Per-sample losses
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Mean reduction
        loss_mean = reduce_loss(losses, reduction='mean')
        assert loss_mean == 2.5
        
        # Sum reduction
        loss_sum = reduce_loss(losses, reduction='sum')
        assert loss_sum == 10.0
        
        # None reduction
        loss_none = reduce_loss(losses, reduction='none')
        assert torch.equal(loss_none, losses)
