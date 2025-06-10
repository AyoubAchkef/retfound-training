"""
Metrics Visualization
====================

Visualization functions for medical metrics and training progress.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

try:
    from sklearn.metrics import roc_curve, auc, calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available for some visualizations")

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> Figure:
    """
    Plot confusion matrix with medical annotations
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Styling
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    if title is None:
        title = 'Confusion Matrix' + (' (Normalized)' if normalize else '')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> Figure:
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn required for ROC curves")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute ROC curve for each class
    n_classes = len(class_names)
    
    for i in range(n_classes):
        # Binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        
        if y_scores.ndim == 1:
            # Binary classification
            y_score_binary = y_scores
        else:
            # Multi-class
            y_score_binary = y_scores[:, i]
        
        # Skip if only one class present
        if len(np.unique(y_true_binary)) < 2:
            continue
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(
            fpr, tpr,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
        )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    
    if title is None:
        title = 'ROC Curves - Multi-class Classification'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    
    return fig


def plot_calibration_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> Figure:
    """
    Plot calibration curves to assess probability reliability
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        class_names: Names of classes
        n_bins: Number of bins for calibration
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn required for calibration curves")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve for each class
    n_classes = len(class_names)
    
    for i in range(min(n_classes, 10)):  # Limit to 10 classes for readability
        # Binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        
        if y_probs.ndim == 1:
            y_prob_binary = y_probs
        else:
            y_prob_binary = y_probs[:, i]
        
        # Skip if only one class present
        if len(np.unique(y_true_binary)) < 2:
            continue
        
        # Compute calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob_binary, n_bins=n_bins
        )
        
        # Plot
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker='o',
            lw=2,
            label=class_names[i]
        )
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    
    # Styling
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    
    if title is None:
        title = 'Calibration Curves - Probability Reliability'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curves saved to {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'accuracy', 'auc'],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Plot training history with multiple metrics
    
    Args:
        history: Dictionary of metric histories
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Find matching keys
        train_key = None
        val_key = None
        
        for key in history.keys():
            if metric in key:
                if 'train' in key:
                    train_key = key
                elif 'val' in key:
                    val_key = key
        
        # Plot training
        if train_key and train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], label='Train', lw=2)
        
        # Plot validation
        if val_key and val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], label='Validation', lw=2)
        
        # Styling
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} History')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    
    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> Figure:
    """
    Plot class distribution in dataset
    
    Args:
        labels: Array of labels
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Bar plot
    bars = ax1.bar(range(len(unique)), counts)
    ax1.set_xticks(range(len(unique)))
    ax1.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=[class_names[i] for i in unique],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Proportions')
    
    if title is None:
        title = 'Dataset Class Distribution'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution saved to {save_path}")
    
    return fig


def create_metrics_dashboard(
    metrics: Dict[str, float],
    class_names: List[str],
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Create comprehensive metrics dashboard
    
    Args:
        metrics: Dictionary of computed metrics
        class_names: Names of classes
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Overall metrics
    ax1 = fig.add_subplot(gs[0, :2])
    overall_metrics = {
        'Accuracy': metrics.get('accuracy', 0),
        'Balanced Acc': metrics.get('balanced_accuracy', 0),
        'Cohen Kappa': metrics.get('cohen_kappa', 0),
        'MCC': metrics.get('matthews_corrcoef', 0),
        'AUC (macro)': metrics.get('auc_macro', 0)
    }
    
    bars = ax1.bar(overall_metrics.keys(), overall_metrics.values())
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar, (name, value) in zip(bars, overall_metrics.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Class-wise sensitivity
    ax2 = fig.add_subplot(gs[1, :])
    sensitivities = []
    sens_labels = []
    
    for class_name in class_names:
        key = f'{class_name}_sensitivity'
        if key in metrics:
            sensitivities.append(metrics[key])
            sens_labels.append(class_name)
    
    if sensitivities:
        bars = ax2.barh(range(len(sensitivities)), sensitivities)
        ax2.set_yticks(range(len(sensitivities)))
        ax2.set_yticklabels(sens_labels)
        ax2.set_xlabel('Sensitivity (%)')
        ax2.set_title('Class-wise Sensitivity', fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add critical threshold lines
        ax2.axvline(x=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.axvline(x=99, color='red', linestyle='--', alpha=0.7, label='99% threshold')
        ax2.legend()
    
    # Medical summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    summary_text = f"""Medical Summary
    
Mean Sensitivity: {metrics.get('mean_sensitivity', 0):.1f}%
Mean Specificity: {metrics.get('mean_specificity', 0):.1f}%

Critical Alerts: {len(metrics.get('critical_alerts', []))}
    """
    
    if 'dr_quadratic_kappa' in metrics:
        summary_text += f"\nDR Quadratic κ: {metrics['dr_quadratic_kappa']:.3f}"
    
    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Medical AI Performance Dashboard', fontsize=18, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics dashboard saved to {save_path}")
    
    return fig