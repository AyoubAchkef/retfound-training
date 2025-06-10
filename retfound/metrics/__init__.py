"""
Metrics Module
=============

Comprehensive metrics for medical image analysis, including
ophthalmology-specific metrics, visualization, and reporting.
"""

from .medical import (
    OphthalmologyMetrics,
    CRITICAL_CONDITIONS,
    compute_sensitivity_specificity,
    compute_ppv_npv,
    compute_confusion_matrix,
    compute_roc_auc,
    compute_cohen_kappa,
    compute_matthews_corrcoef
)

from .visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_calibration_curves,
    plot_training_history,
    plot_class_distribution,
    create_metrics_dashboard
)

from .reports import (
    generate_clinical_report,
    generate_training_report,
    generate_model_card,
    create_latex_report,
    export_metrics_to_csv
)

__all__ = [
    # Medical metrics
    'OphthalmologyMetrics',
    'CRITICAL_CONDITIONS',
    'compute_sensitivity_specificity',
    'compute_ppv_npv',
    'compute_confusion_matrix',
    'compute_roc_auc',
    'compute_cohen_kappa',
    'compute_matthews_corrcoef',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_calibration_curves',
    'plot_training_history',
    'plot_class_distribution',
    'create_metrics_dashboard',
    
    # Reports
    'generate_clinical_report',
    'generate_training_report',
    'generate_model_card',
    'create_latex_report',
    'export_metrics_to_csv'
]