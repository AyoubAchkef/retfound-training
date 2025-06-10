"""
Metrics Tracking Callbacks
=========================

Implements callbacks for tracking and computing metrics during training.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F

from .base import Callback

logger = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """
    Callback for tracking and computing metrics during training
    """
    
    def __init__(
        self,
        metrics: Optional[Dict[str, Callable]] = None,
        compute_on_train: bool = True,
        compute_on_val: bool = True,
        reset_on_epoch: bool = True
    ):
        """
        Initialize metrics callback
        
        Args:
            metrics: Dictionary of metric name to metric function
            compute_on_train: Whether to compute metrics on training data
            compute_on_val: Whether to compute metrics on validation data
            reset_on_epoch: Whether to reset metrics at epoch start
        """
        super().__init__()
        self.metrics = metrics or {}
        self.compute_on_train = compute_on_train
        self.compute_on_val = compute_on_val
        self.reset_on_epoch = reset_on_epoch
        
        # Storage for metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        
        # Accumulate predictions and targets
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Reset metrics at epoch start"""
        if self.reset_on_epoch:
            self.train_preds = []
            self.train_targets = []
            self.val_preds = []
            self.val_targets = []
            self.batch_metrics = defaultdict(list)
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float,
        outputs: Any,
        targets: Any
    ):
        """Accumulate predictions for metric computation"""
        if trainer.model.training and self.compute_on_train:
            # Store predictions and targets
            with torch.no_grad():
                preds = outputs.detach()
                if preds.dim() > 1 and preds.size(1) > 1:  # Multi-class
                    preds = F.softmax(preds, dim=1)
                self.train_preds.append(preds.cpu())
                self.train_targets.append(targets.cpu())
        
        # Store batch loss
        self.batch_metrics['loss'].append(loss)
    
    def on_validation_batch_end(
        self,
        trainer,
        batch_idx: int,
        outputs: Any,
        targets: Any
    ):
        """Accumulate validation predictions"""
        if self.compute_on_val:
            with torch.no_grad():
                preds = outputs.detach()
                if preds.dim() > 1 and preds.size(1) > 1:  # Multi-class
                    preds = F.softmax(preds, dim=1)
                self.val_preds.append(preds.cpu())
                self.val_targets.append(targets.cpu())
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Compute and store epoch metrics"""
        # Compute training metrics
        if self.compute_on_train and self.train_preds:
            train_preds = torch.cat(self.train_preds, dim=0)
            train_targets = torch.cat(self.train_targets, dim=0)
            
            computed_train_metrics = self._compute_metrics(train_preds, train_targets)
            
            # Add to trainer metrics
            for name, value in computed_train_metrics.items():
                train_metrics[f'{name}'] = value
                self.train_metrics[name].append(value)
        
        # Compute validation metrics
        if self.compute_on_val and self.val_preds:
            val_preds = torch.cat(self.val_preds, dim=0)
            val_targets = torch.cat(self.val_targets, dim=0)
            
            computed_val_metrics = self._compute_metrics(val_preds, val_targets)
            
            # Add to trainer metrics
            for name, value in computed_val_metrics.items():
                val_metrics[f'{name}'] = value
                self.val_metrics[name].append(value)
        
        # Add average batch metrics
        for name, values in self.batch_metrics.items():
            if values:
                train_metrics[f'avg_{name}'] = np.mean(values)
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute all registered metrics"""
        results = {}
        
        for name, metric_fn in self.metrics.items():
            try:
                value = metric_fn(predictions, targets)
                results[name] = float(value)
            except Exception as e:
                logger.warning(f"Failed to compute metric {name}: {e}")
                results[name] = 0.0
        
        return results


class AccuracyMetric:
    """Simple accuracy metric"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        correct = (pred_classes == targets).float().sum()
        total = targets.size(0)
        
        return (correct / total * 100).item()


class AUCMetric:
    """AUC-ROC metric for binary and multi-class"""
    
    def __init__(self, average: str = 'macro', multi_class: str = 'ovr'):
        self.average = average
        self.multi_class = multi_class
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            
            # Convert to numpy
            preds_np = predictions.numpy()
            targets_np = targets.numpy()
            
            # Binary classification
            if predictions.size(1) == 2:
                return roc_auc_score(targets_np, preds_np[:, 1])
            
            # Multi-class
            return roc_auc_score(
                targets_np,
                preds_np,
                average=self.average,
                multi_class=self.multi_class
            )
            
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")
            return 0.5


class F1Metric:
    """F1 score metric"""
    
    def __init__(self, average: str = 'macro'):
        self.average = average
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        try:
            from sklearn.metrics import f1_score
            
            # Get predicted classes
            if predictions.dim() > 1:
                pred_classes = predictions.argmax(dim=1).numpy()
            else:
                pred_classes = (predictions > 0.5).long().numpy()
            
            targets_np = targets.numpy()
            
            return f1_score(targets_np, pred_classes, average=self.average)
            
        except Exception as e:
            logger.warning(f"F1 computation failed: {e}")
            return 0.0


class BalancedAccuracyMetric:
    """Balanced accuracy metric"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        try:
            from sklearn.metrics import balanced_accuracy_score
            
            # Get predicted classes
            if predictions.dim() > 1:
                pred_classes = predictions.argmax(dim=1).numpy()
            else:
                pred_classes = (predictions > 0.5).long().numpy()
            
            targets_np = targets.numpy()
            
            return balanced_accuracy_score(targets_np, pred_classes) * 100
            
        except Exception as e:
            logger.warning(f"Balanced accuracy computation failed: {e}")
            return 0.0


class CohenKappaMetric:
    """Cohen's Kappa metric"""
    
    def __init__(self, weights: Optional[str] = None):
        self.weights = weights
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        try:
            from sklearn.metrics import cohen_kappa_score
            
            # Get predicted classes
            if predictions.dim() > 1:
                pred_classes = predictions.argmax(dim=1).numpy()
            else:
                pred_classes = (predictions > 0.5).long().numpy()
            
            targets_np = targets.numpy()
            
            return cohen_kappa_score(targets_np, pred_classes, weights=self.weights)
            
        except Exception as e:
            logger.warning(f"Cohen's Kappa computation failed: {e}")
            return 0.0


class MatthewsCorrCoefMetric:
    """Matthews Correlation Coefficient metric"""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        try:
            from sklearn.metrics import matthews_corrcoef
            
            # Get predicted classes
            if predictions.dim() > 1:
                pred_classes = predictions.argmax(dim=1).numpy()
            else:
                pred_classes = (predictions > 0.5).long().numpy()
            
            targets_np = targets.numpy()
            
            return matthews_corrcoef(targets_np, pred_classes)
            
        except Exception as e:
            logger.warning(f"MCC computation failed: {e}")
            return 0.0


class ClassWiseMetricsCallback(Callback):
    """
    Callback for tracking class-wise metrics
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        metrics_to_track: List[str] = ['sensitivity', 'specificity', 'ppv', 'npv']
    ):
        """
        Initialize class-wise metrics callback
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
            metrics_to_track: Which metrics to compute per class
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.metrics_to_track = metrics_to_track
        
        # Storage
        self.val_preds = []
        self.val_targets = []
    
    def on_validation_batch_end(
        self,
        trainer,
        batch_idx: int,
        outputs: Any,
        targets: Any
    ):
        """Accumulate validation predictions"""
        with torch.no_grad():
            preds = outputs.detach()
            if preds.dim() > 1 and preds.size(1) > 1:
                preds = F.softmax(preds, dim=1)
            self.val_preds.append(preds.cpu())
            self.val_targets.append(targets.cpu())
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Compute class-wise metrics"""
        if not self.val_preds:
            return
        
        # Concatenate all predictions
        all_preds = torch.cat(self.val_preds, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)
        
        # Get predicted classes
        pred_classes = all_preds.argmax(dim=1)
        
        # Compute metrics for each class
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            
            # Binary masks for this class
            true_mask = (all_targets == class_idx)
            pred_mask = (pred_classes == class_idx)
            
            # Calculate confusion matrix components
            tp = (true_mask & pred_mask).sum().item()
            tn = (~true_mask & ~pred_mask).sum().item()
            fp = (~true_mask & pred_mask).sum().item()
            fn = (true_mask & ~pred_mask).sum().item()
            
            # Calculate metrics
            if 'sensitivity' in self.metrics_to_track:
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                val_metrics[f'{class_name}_sensitivity'] = sensitivity * 100
            
            if 'specificity' in self.metrics_to_track:
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                val_metrics[f'{class_name}_specificity'] = specificity * 100
            
            if 'ppv' in self.metrics_to_track:
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                val_metrics[f'{class_name}_ppv'] = ppv * 100
            
            if 'npv' in self.metrics_to_track:
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                val_metrics[f'{class_name}_npv'] = npv * 100
            
            # F1 score
            if 'f1' in self.metrics_to_track:
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                val_metrics[f'{class_name}_f1'] = f1
        
        # Clear stored predictions
        self.val_preds = []
        self.val_targets = []


def create_standard_metrics() -> Dict[str, Callable]:
    """Create standard set of metrics"""
    return {
        'accuracy': AccuracyMetric(),
        'balanced_accuracy': BalancedAccuracyMetric(),
        'auc_macro': AUCMetric(average='macro'),
        'f1_macro': F1Metric(average='macro'),
        'cohen_kappa': CohenKappaMetric(),
        'mcc': MatthewsCorrCoefMetric()
    }