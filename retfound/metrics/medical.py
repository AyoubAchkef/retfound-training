"""
Medical-specific metrics for ophthalmology tasks.
Supports both v4.0 (22 classes) and v6.1 (28 classes) datasets.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, 
    cohen_kappa_score, 
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.preprocessing import label_binarize

from ..core.constants import (
    UNIFIED_CLASS_NAMES,
    CRITICAL_CONDITIONS,
    FUNDUS_CLASS_NAMES,
    OCT_CLASS_NAMES
)


class OphthalmologyMetrics:
    """Comprehensive metrics for ophthalmology classification tasks."""
    
    def __init__(
        self,
        num_classes: int = 28,
        dataset_version: str = "v6.1",
        modality: Optional[str] = None,
        monitor_critical: bool = True
    ):
        """
        Initialize ophthalmology metrics calculator.
        
        Args:
            num_classes: Number of classes (22 for v4.0, 28 for v6.1)
            dataset_version: Dataset version ("v4.0" or "v6.1")
            modality: Optional modality filter ("fundus", "oct", or None for both)
            monitor_critical: Whether to monitor critical conditions
        """
        self.num_classes = num_classes
        self.dataset_version = dataset_version
        self.modality = modality
        self.monitor_critical = monitor_critical
        
        # Set class names based on version
        if dataset_version == "v6.1":
            self.class_names = UNIFIED_CLASS_NAMES
            if modality == "fundus":
                self.active_classes = list(range(18))  # Classes 0-17
            elif modality == "oct":
                self.active_classes = list(range(18, 28))  # Classes 18-27
            else:
                self.active_classes = list(range(28))  # All classes
        else:
            # For backward compatibility with v4.0
            self.class_names = UNIFIED_CLASS_NAMES[:22]
            self.active_classes = list(range(22))
            
        # Critical conditions indices
        self.critical_indices = self._get_critical_indices()
        
        # Initialize metric storage
        self.reset()
    
    def _get_critical_indices(self) -> List[int]:
        """Get indices of critical conditions for monitoring."""
        if not self.monitor_critical or self.dataset_version != "v6.1":
            return []
            
        critical_indices = []
        for condition, info in CRITICAL_CONDITIONS.items():
            if condition in self.class_names:
                idx = self.class_names.index(condition)
                if idx in self.active_classes:
                    critical_indices.append(idx)
        return critical_indices
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_metadata = []
    
    def update(
        self, 
        preds: torch.Tensor, 
        labels: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            preds: Predicted class indices [batch_size]
            labels: True class indices [batch_size]
            probs: Optional prediction probabilities [batch_size, num_classes]
            metadata: Optional metadata for each sample
        """
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
            
        if metadata is not None:
            self.all_metadata.extend(metadata)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if len(self.all_preds) == 0:
            return {}
            
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = self._compute_accuracy(preds, labels)
        metrics['balanced_accuracy'] = self._compute_balanced_accuracy(preds, labels)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, labels=self.active_classes, average=None, zero_division=0
        )
        
        # Store per-class metrics
        for i, class_idx in enumerate(self.active_classes):
            class_name = self.class_names[class_idx]
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = support[i]
        
        # Weighted and macro averages
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        metrics['precision_weighted'] = np.average(precision, weights=support)
        metrics['recall_weighted'] = np.average(recall, weights=support)
        metrics['f1_weighted'] = np.average(f1, weights=support)
        
        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(labels, preds)
        
        # Quadratic Kappa for DR grading (if applicable)
        if self._has_dr_classes(labels):
            metrics['quadratic_kappa'] = self._compute_quadratic_kappa_dr(preds, labels)
        
        # AUC-ROC (if probabilities available)
        if len(self.all_probs) > 0:
            auc_metrics = self._compute_auc_metrics(np.array(self.all_probs), labels)
            metrics.update(auc_metrics)
        
        # Critical conditions monitoring
        if self.monitor_critical and len(self.critical_indices) > 0:
            critical_metrics = self._compute_critical_metrics(preds, labels)
            metrics.update(critical_metrics)
        
        # Modality-specific metrics (v6.1 only)
        if self.dataset_version == "v6.1" and self.modality is None:
            modality_metrics = self._compute_modality_metrics(preds, labels)
            metrics.update(modality_metrics)
        
        return metrics
    
    def _compute_accuracy(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute overall accuracy."""
        return np.mean(preds == labels)
    
    def _compute_balanced_accuracy(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute balanced accuracy across all classes."""
        cm = confusion_matrix(labels, preds, labels=self.active_classes)
        per_class_accuracy = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        return np.mean(per_class_accuracy)
    
    def _has_dr_classes(self, labels: np.ndarray) -> bool:
        """Check if DR classes are present in labels."""
        dr_classes = ["DR_Mild", "DR_Moderate", "DR_Severe", "DR_Proliferative"]
        dr_indices = []
        
        for dr_class in dr_classes:
            if dr_class in self.class_names:
                idx = self.class_names.index(dr_class)
                if idx in self.active_classes:
                    dr_indices.append(idx)
        
        return len(dr_indices) > 0 and any(label in dr_indices for label in labels)
    
    def _compute_quadratic_kappa_dr(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute quadratic kappa specifically for DR grading."""
        # Map predictions to DR grades (0-4)
        dr_mapping = {
            "Normal_Fundus": 0,
            "DR_Mild": 1,
            "DR_Moderate": 2,
            "DR_Severe": 3,
            "DR_Proliferative": 4
        }
        
        # Create reverse mapping from class index to DR grade
        idx_to_grade = {}
        for class_name, grade in dr_mapping.items():
            if class_name in self.class_names:
                idx = self.class_names.index(class_name)
                idx_to_grade[idx] = grade
        
        # Filter and map to DR grades
        dr_mask = np.array([label in idx_to_grade for label in labels])
        if dr_mask.sum() == 0:
            return 0.0
            
        dr_preds = np.array([idx_to_grade.get(p, -1) for p in preds[dr_mask]])
        dr_labels = np.array([idx_to_grade[l] for l in labels[dr_mask]])
        
        # Remove invalid predictions
        valid_mask = dr_preds >= 0
        if valid_mask.sum() == 0:
            return 0.0
            
        return cohen_kappa_score(
            dr_labels[valid_mask], 
            dr_preds[valid_mask], 
            weights='quadratic'
        )
    
    def _compute_auc_metrics(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute AUC-ROC metrics."""
        metrics = {}
        
        # Binarize labels for multi-class AUC
        labels_bin = label_binarize(labels, classes=self.active_classes)
        
        # Filter probabilities to active classes only
        if len(self.active_classes) < self.num_classes:
            probs = probs[:, self.active_classes]
        
        # Per-class AUC
        for i, class_idx in enumerate(self.active_classes):
            if labels_bin[:, i].sum() > 0:  # Class present in labels
                try:
                    auc = roc_auc_score(labels_bin[:, i], probs[:, i])
                    class_name = self.class_names[class_idx]
                    metrics[f'auc_{class_name}'] = auc
                except:
                    pass
        
        # Macro and weighted AUC
        try:
            # Only compute for classes present in labels
            present_classes = [i for i in range(len(self.active_classes)) 
                             if labels_bin[:, i].sum() > 0]
            if len(present_classes) > 1:
                metrics['auc_macro'] = roc_auc_score(
                    labels_bin[:, present_classes], 
                    probs[:, present_classes], 
                    average='macro'
                )
                metrics['auc_weighted'] = roc_auc_score(
                    labels_bin[:, present_classes], 
                    probs[:, present_classes], 
                    average='weighted'
                )
        except:
            pass
        
        return metrics
    
    def _compute_critical_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute metrics for critical conditions."""
        metrics = {}
        
        for idx in self.critical_indices:
            condition_name = self.class_names[idx]
            condition_mask = labels == idx
            
            if condition_mask.sum() > 0:
                # True positives and false negatives
                tp = ((preds == idx) & condition_mask).sum()
                fn = ((preds != idx) & condition_mask).sum()
                
                sensitivity = tp / (tp + fn + 1e-10)
                
                # Get threshold from constants
                threshold = CRITICAL_CONDITIONS.get(condition_name, {}).get('min_sensitivity', 0.9)
                
                metrics[f'critical_{condition_name}_sensitivity'] = sensitivity
                metrics[f'critical_{condition_name}_meets_threshold'] = float(sensitivity >= threshold)
                
                # Additional critical metrics
                tn = ((preds != idx) & ~condition_mask).sum()
                fp = ((preds == idx) & ~condition_mask).sum()
                
                specificity = tn / (tn + fp + 1e-10)
                metrics[f'critical_{condition_name}_specificity'] = specificity
        
        # Overall critical performance
        if len(self.critical_indices) > 0:
            critical_sensitivities = []
            for idx in self.critical_indices:
                condition_name = self.class_names[idx]
                if f'critical_{condition_name}_sensitivity' in metrics:
                    critical_sensitivities.append(metrics[f'critical_{condition_name}_sensitivity'])
            
            if critical_sensitivities:
                metrics['critical_avg_sensitivity'] = np.mean(critical_sensitivities)
                metrics['critical_min_sensitivity'] = np.min(critical_sensitivities)
        
        return metrics
    
    def _compute_modality_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute separate metrics for fundus and OCT modalities."""
        metrics = {}
        
        # Fundus metrics (classes 0-17)
        fundus_mask = labels < 18
        if fundus_mask.sum() > 0:
            fundus_preds = preds[fundus_mask]
            fundus_labels = labels[fundus_mask]
            metrics['fundus_accuracy'] = np.mean(fundus_preds == fundus_labels)
            
            # Only compute if predictions are in fundus range
            fundus_correct_range = fundus_preds < 18
            if fundus_correct_range.sum() > 0:
                metrics['fundus_correct_modality'] = fundus_correct_range.mean()
        
        # OCT metrics (classes 18-27)
        oct_mask = labels >= 18
        if oct_mask.sum() > 0:
            oct_preds = preds[oct_mask]
            oct_labels = labels[oct_mask]
            metrics['oct_accuracy'] = np.mean(oct_preds == oct_labels)
            
            # Only compute if predictions are in OCT range
            oct_correct_range = oct_preds >= 18
            if oct_correct_range.sum() > 0:
                metrics['oct_correct_modality'] = oct_correct_range.mean()
        
        return metrics
    
    def get_classification_report(self, output_dict: bool = False) -> Union[str, Dict]:
        """
        Get detailed classification report.
        
        Args:
            output_dict: If True, return as dictionary
            
        Returns:
            Classification report as string or dictionary
        """
        if len(self.all_preds) == 0:
            return {} if output_dict else "No predictions available"
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Create target names for active classes
        target_names = [self.class_names[i] for i in self.active_classes]
        
        return classification_report(
            labels, preds,
            labels=self.active_classes,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def get_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            normalize: {'true', 'pred', 'all'} or None
            
        Returns:
            Confusion matrix
        """
        if len(self.all_preds) == 0:
            return np.zeros((len(self.active_classes), len(self.active_classes)))
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        cm = confusion_matrix(labels, preds, labels=self.active_classes)
        
        if normalize:
            if normalize == 'true':
                cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
            elif normalize == 'pred':
                cm = cm.astype('float') / (cm.sum(axis=0, keepdims=True) + 1e-10)
            elif normalize == 'all':
                cm = cm.astype('float') / (cm.sum() + 1e-10)
        
        return cm


# Standalone utility functions for backward compatibility and direct use
def compute_sensitivity_specificity(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    pos_label: int = 1
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        pos_label: Label of positive class
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1-pos_label, pos_label])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
    else:
        sensitivity = specificity = 0.0
    
    return sensitivity, specificity


def compute_ppv_npv(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    pos_label: int = 1
) -> Tuple[float, float]:
    """
    Compute positive predictive value (precision) and negative predictive value.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        pos_label: Label of positive class
        
    Returns:
        Tuple of (PPV, NPV)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1-pos_label, pos_label])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        ppv = tp / (tp + fp + 1e-10)  # Precision
        npv = tn / (tn + fn + 1e-10)  # Negative predictive value
    else:
        ppv = npv = 0.0
    
    return ppv, npv


def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: Optional[List[int]] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix with optional normalization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to include in matrix
        normalize: {'true', 'pred', 'all'} or None
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        if normalize == 'true':
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        elif normalize == 'pred':
            cm = cm.astype('float') / (cm.sum(axis=0, keepdims=True) + 1e-10)
        elif normalize == 'all':
            cm = cm.astype('float') / (cm.sum() + 1e-10)
    
    return cm


def compute_roc_auc(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    multi_class: str = 'ovr',
    average: str = 'macro'
) -> float:
    """
    Compute ROC AUC score.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        multi_class: {'ovr', 'ovo'} for multiclass
        average: {'macro', 'weighted'} for multiclass
        
    Returns:
        ROC AUC score
    """
    try:
        return roc_auc_score(
            y_true, y_score, 
            multi_class=multi_class, 
            average=average
        )
    except ValueError:
        return 0.0


def compute_cohen_kappa(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    weights: Optional[str] = None
) -> float:
    """
    Compute Cohen's kappa coefficient.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        weights: {'linear', 'quadratic'} or None
        
    Returns:
        Cohen's kappa coefficient
    """
    return cohen_kappa_score(y_true, y_pred, weights=weights)


def compute_matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Matthews correlation coefficient.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Matthews correlation coefficient
    """
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)
