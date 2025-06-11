"""
Loss Functions for RETFound
==========================

Implements various loss functions including label smoothing,
focal loss, and asymmetric loss for handling class imbalance.
"""

import logging
from typing import Optional, Union, Tuple, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import RETFoundConfig
from ..core.registry import Registry

logger = logging.getLogger(__name__)

# Loss registry
LOSS_REGISTRY = Registry("losses")


def register_loss(name: str):
    """Decorator to register a loss function"""
    def decorator(cls):
        LOSS_REGISTRY.register(name, cls)
        return cls
    return decorator


@register_loss("label_smoothing")
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing
    
    Label smoothing helps prevent overconfidence and improves generalization
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None
    ):
        """
        Initialize label smoothing loss
        
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing)
            reduction: Reduction method ('mean', 'sum', 'none')
            weight: Class weights
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
        self.weight = weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate label smoothing cross entropy loss
        
        Args:
            pred: Predictions (logits) [B, C]
            target: Target labels [B]
            
        Returns:
            Loss value
        """
        # Log softmax
        pred = pred.log_softmax(dim=-1)
        
        # Get true class log probabilities
        nll_loss = -pred.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        # Get smoothed loss (KL divergence with uniform distribution)
        smooth_loss = -pred.mean(dim=-1)
        
        # Combine losses
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss * weight
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@register_loss("focal")
class FocalLoss(nn.Module):
    """
    Focal loss for addressing extreme class imbalance
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize focal loss
        
        Args:
            alpha: Class weights (scalar or tensor)
            gamma: Focusing parameter
            reduction: Reduction method
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            inputs: Predictions (logits) [B, C]
            targets: Target labels [B]
            
        Returns:
            Loss value
        """
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Use smoothed targets for CE calculation
            ce_loss = F.cross_entropy(inputs, smooth_targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Tensor of class weights
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


@register_loss("asymmetric")
class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss for handling class imbalance with different
    penalties for false positives and false negatives
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Initialize asymmetric loss
        
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping value
            reduction: Reduction method
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate asymmetric loss
        
        Args:
            inputs: Predictions (logits) [B, C]
            targets: Target labels [B]
            
        Returns:
            Loss value
        """
        num_classes = inputs.size(1)
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Clip probabilities
        if self.clip > 0:
            probs = torch.clamp(probs, self.clip, 1.0 - self.clip)
        
        # Calculate positive and negative losses
        pos_loss = targets_one_hot * torch.log(probs)
        neg_loss = (1 - targets_one_hot) * torch.log(1 - probs)
        
        # Apply asymmetric focusing
        pos_loss = pos_loss * torch.pow(1 - probs, self.gamma_pos)
        neg_loss = neg_loss * torch.pow(probs, self.gamma_neg)
        
        # Combine losses
        loss = -(pos_loss + neg_loss)
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@register_loss("weighted_ce")
class WeightedCrossEntropy(nn.Module):
    """
    Cross entropy with automatic class weighting
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize weighted cross entropy
        
        Args:
            weight: Class weights
            label_smoothing: Label smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                reduction=reduction,
                weight=weight
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                reduction=reduction
            )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate weighted cross entropy loss"""
        return self.criterion(inputs, targets)


class MixupCutmixCriterion(nn.Module):
    """
    Loss function for MixUp/CutMix augmented data
    """
    
    def __init__(self, base_criterion: nn.Module):
        """
        Initialize MixUp/CutMix criterion
        
        Args:
            base_criterion: Base loss function to use
        """
        super().__init__()
        self.base_criterion = base_criterion
    
    def forward(
        self,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Calculate loss for mixed samples
        
        Args:
            pred: Predictions [B, C]
            y_a: First set of labels [B]
            y_b: Second set of labels [B]
            lam: Mixing coefficient
            
        Returns:
            Mixed loss
        """
        return (
            lam * self.base_criterion(pred, y_a) +
            (1 - lam) * self.base_criterion(pred, y_b)
        )


def calculate_class_weights(
    labels: list,
    num_classes: int,
    weight_type: str = 'inverse_frequency',
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: List of labels
        num_classes: Number of classes
        weight_type: Type of weighting ('inverse_frequency', 'effective_number')
        device: Device to place weights on
        
    Returns:
        Class weights tensor
    """
    # Count class frequencies
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Initialize weights
    weights = torch.ones(num_classes, device=device)
    
    if weight_type == 'inverse_frequency':
        # Inverse class frequency
        for class_id, count in class_counts.items():
            if class_id < num_classes:
                weights[class_id] = total_samples / (num_classes * count)
    
    elif weight_type == 'effective_number':
        # Effective number of samples
        # Reference: https://arxiv.org/abs/1901.05555
        beta = 0.9999
        for class_id, count in class_counts.items():
            if class_id < num_classes:
                effective_num = (1 - beta ** count) / (1 - beta)
                weights[class_id] = 1.0 / effective_num
    
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Log class weights
    logger.info("Class weights:")
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        logger.info(f"  Class {i}: weight={weights[i]:.3f}, count={count}")
    
    return weights


def create_loss_function(
    config: RETFoundConfig,
    train_dataset: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Create loss function based on configuration
    
    Args:
        config: Configuration object
        train_dataset: Training dataset for calculating class weights
        device: Device to place loss function on
        
    Returns:
        Loss function
    """
    # Calculate class weights if needed
    class_weights = None
    
    if config.use_class_weights and train_dataset is not None:
        if hasattr(train_dataset, 'targets'):
            labels = train_dataset.targets
        elif hasattr(train_dataset, 'get_labels'):
            labels = train_dataset.get_labels()
        else:
            labels = []
            logger.warning("Could not extract labels from dataset")
        
        if labels:
            class_weights = calculate_class_weights(
                labels,
                config.num_classes,
                weight_type='inverse_frequency',
                device=device
            )
            
            # Check for extreme imbalance
            imbalance_ratio = class_weights.max() / class_weights.min()
            logger.info(f"Class weight ratio: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 10 and not config.use_focal_loss:
                logger.warning(
                    f"High class imbalance detected ({imbalance_ratio:.1f}:1). "
                    "Consider using focal loss."
                )
    
    # Create loss function
    if config.use_focal_loss:
        logger.info("Using Focal Loss")
        loss_fn = FocalLoss(
            alpha=class_weights,
            gamma=config.focal_gamma,
            reduction='mean',
            label_smoothing=config.label_smoothing
        )
    
    elif config.label_smoothing > 0:
        logger.info(f"Using Label Smoothing CE (smoothing={config.label_smoothing})")
        loss_fn = LabelSmoothingCrossEntropy(
            smoothing=config.label_smoothing,
            weight=class_weights
        )
    
    else:
        logger.info("Using Weighted Cross Entropy")
        loss_fn = WeightedCrossEntropy(
            weight=class_weights,
            label_smoothing=0.0
        )
    
    # Move to device
    loss_fn = loss_fn.to(device)
    
    return loss_fn


def create_custom_loss(
    loss_name: str,
    **kwargs
) -> nn.Module:
    """
    Create a custom loss function from registry
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Loss function arguments
        
    Returns:
        Loss function instance
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{loss_name}' not found. "
            f"Available losses: {list(LOSS_REGISTRY.keys())}"
        )
    
    loss_class = LOSS_REGISTRY.get(loss_name)
    return loss_class(**kwargs)
