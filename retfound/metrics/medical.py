"""
Medical Metrics
==============

Specialized metrics for medical image analysis with focus on ophthalmology.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, auc, cohen_kappa_score,
        confusion_matrix, classification_report, f1_score,
        balanced_accuracy_score, matthews_corrcoef, roc_curve,
        calibration_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, some metrics will be disabled")

logger = logging.getLogger(__name__)


# Critical conditions requiring high sensitivity
CRITICAL_CONDITIONS = {
    'RAO': {'min_sensitivity': 0.99, 'reason': 'Retinal artery occlusion - Emergency'},
    'RVO': {'min_sensitivity': 0.97, 'reason': 'Retinal vein occlusion - Urgent'},
    'Retinal_Detachment': {'min_sensitivity': 0.99, 'reason': 'Surgical emergency'},
    'CNV': {'min_sensitivity': 0.98, 'reason': 'Risk of rapid vision loss'},
    'DR_Proliferative': {'min_sensitivity': 0.98, 'reason': 'Risk of vitreous hemorrhage'},
    'DME': {'min_sensitivity': 0.95, 'reason': 'Leading cause of vision loss in diabetics'},
    'Glaucoma_Positive': {'min_sensitivity': 0.95, 'reason': 'Irreversible vision loss'}
}


@dataclass
class MetricResult:
    """Container for metric results"""
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    details: Optional[Dict[str, Any]] = None


class OphthalmologyMetrics:
    """
    Comprehensive metrics for ophthalmology classification
    
    Includes standard metrics plus medical-specific evaluations.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, probabilities: Optional[np.ndarray] = None):
        """
        Update metrics with batch results
        
        Args:
            predictions: Predicted classes
            targets: True classes
            probabilities: Prediction probabilities (optional)
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if probabilities is not None:
            self.probabilities.extend(probabilities)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive medical metrics"""
        if not self.predictions:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities) if self.probabilities else None
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = (y_pred == y_true).mean() * 100
        
        if SKLEARN_AVAILABLE:
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred) * 100
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Per-class metrics
        class_metrics = self._compute_class_metrics(y_true, y_pred, y_prob)
        metrics.update(class_metrics)
        
        # Overall medical metrics
        medical_metrics = self._compute_medical_metrics(y_true, y_pred, y_prob)
        metrics.update(medical_metrics)
        
        # Check critical conditions
        critical_alerts = self._check_critical_conditions(metrics)
        if critical_alerts:
            metrics['critical_alerts'] = critical_alerts
        
        return metrics
    
    def _compute_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute per-class metrics"""
        metrics = {}
        
        for i in range(self.num_classes):
            if i >= len(self.class_names):
                class_name = f"Class_{i}"
            else:
                class_name = self.class_names[i]
            
            # Binary metrics for each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Confusion matrix components
            tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
            tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
            fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
            fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
            
            # Sensitivity (Recall)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{class_name}_sensitivity'] = sensitivity * 100
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'{class_name}_specificity'] = specificity * 100
            
            # PPV (Precision)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics[f'{class_name}_ppv'] = ppv * 100
            
            # NPV
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics[f'{class_name}_npv'] = npv * 100
            
            # F1 Score
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            metrics[f'{class_name}_f1'] = f1
            
            # AUC if probabilities available
            if y_prob is not None and y_prob.shape[1] > i and SKLEARN_AVAILABLE:
                y_prob_binary = y_prob[:, i]
                try:
                    if len(np.unique(y_true_binary)) > 1:
                        auc_score = roc_auc_score(y_true_binary, y_prob_binary)
                    else:
                        auc_score = 0.5
                except:
                    auc_score = 0.5
                metrics[f'{class_name}_auc'] = auc_score
        
        return metrics
    
    def _compute_medical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute medical-specific metrics"""
        metrics = {}
        
        # Mean sensitivity and specificity
        sensitivities = []
        specificities = []
        
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            sens_key = f'{class_name}_sensitivity'
            spec_key = f'{class_name}_specificity'
            
            if sens_key in metrics:
                sensitivities.append(metrics[sens_key])
            if spec_key in metrics:
                specificities.append(metrics[spec_key])
        
        if sensitivities:
            metrics['mean_sensitivity'] = np.mean(sensitivities)
        if specificities:
            metrics['mean_specificity'] = np.mean(specificities)
        
        # Overall AUC
        if y_prob is not None and SKLEARN_AVAILABLE:
            try:
                if self.num_classes > 2:
                    metrics['auc_macro'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                    metrics['auc_weighted'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
                else:
                    metrics['auc_macro'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc_weighted'] = metrics['auc_macro']
            except:
                metrics['auc_macro'] = 0.5
                metrics['auc_weighted'] = 0.5
        
        # Diabetic Retinopathy specific: Quadratic Kappa
        dr_classes = [i for i, name in enumerate(self.class_names)
                     if any(dr in name.lower() for dr in ['dr_', 'diabetic', 'mild', 'moderate', 'severe', 'proliferative'])]
        
        if len(dr_classes) >= 4 and SKLEARN_AVAILABLE:
            dr_mask = np.isin(y_true, dr_classes) | np.isin(y_pred, dr_classes)
            if dr_mask.any():
                try:
                    metrics['dr_quadratic_kappa'] = cohen_kappa_score(
                        y_true[dr_mask], y_pred[dr_mask], weights='quadratic'
                    )
                except:
                    pass
        
        return metrics
    
    def _check_critical_conditions(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check performance on critical conditions"""
        critical_alerts = []
        
        for condition, requirements in CRITICAL_CONDITIONS.items():
            for i, class_name in enumerate(self.class_names[:self.num_classes]):
                if condition in class_name:
                    sensitivity = metrics.get(f'{class_name}_sensitivity', 0)
                    if sensitivity < requirements['min_sensitivity'] * 100:
                        critical_alerts.append({
                            'class': class_name,
                            'current_sensitivity': sensitivity,
                            'required_sensitivity': requirements['min_sensitivity'] * 100,
                            'reason': requirements['reason']
                        })
        
        return critical_alerts
    
    def confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix"""
        if not self.predictions or not SKLEARN_AVAILABLE:
            return None
        
        return confusion_matrix(self.targets, self.predictions)
    
    def classification_report(self) -> Optional[str]:
        """Get classification report"""
        if not self.predictions or not SKLEARN_AVAILABLE:
            return None
        
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names[:self.num_classes]
        )


def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute sensitivity and specificity
    
    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def compute_ppv_npv(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute positive and negative predictive values
    
    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        
    Returns:
        Tuple of (PPV, NPV)
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return ppv, npv


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix"""
    if SKLEARN_AVAILABLE:
        return confusion_matrix(y_true, y_pred)
    else:
        # Simple implementation
        n_classes = max(y_true.max(), y_pred.max()) + 1
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        return cm


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray, 
                   average: str = 'macro') -> float:
    """Compute ROC AUC score"""
    if not SKLEARN_AVAILABLE:
        return 0.5
    
    try:
        if y_score.ndim == 1:
            # Binary classification
            return roc_auc_score(y_true, y_score)
        else:
            # Multi-class
            return roc_auc_score(y_true, y_score, multi_class='ovr', average=average)
    except:
        return 0.5


def compute_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, 
                       weights: Optional[str] = None) -> float:
    """Compute Cohen's Kappa score"""
    if not SKLEARN_AVAILABLE:
        return 0.0
    
    return cohen_kappa_score(y_true, y_pred, weights=weights)


def compute_matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Matthews correlation coefficient"""
    if not SKLEARN_AVAILABLE:
        return 0.0
    
    return matthews_corrcoef(y_true, y_pred)