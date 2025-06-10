"""
Metrics Reports
==============

Generate comprehensive reports for medical AI models.
"""

import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def generate_clinical_report(
    metrics: Dict[str, Any],
    model_info: Dict[str, Any],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    include_recommendations: bool = True
) -> str:
    """
    Generate comprehensive clinical evaluation report
    
    Args:
        metrics: Computed metrics dictionary
        model_info: Model information (name, version, etc.)
        class_names: Names of classes
        output_path: Path to save report
        include_recommendations: Whether to include clinical recommendations
        
    Returns:
        Report as string
    """
    report = []
    
    # Header
    report.append("=" * 70)
    report.append("CLINICAL EVALUATION REPORT - RETFOUND")
    report.append("=" * 70)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: {model_info.get('name', 'Unknown')}")
    report.append(f"Version: {model_info.get('version', 'Unknown')}")
    report.append(f"Dataset: {model_info.get('dataset', 'Unknown')}")
    report.append("")
    
    # Overall Performance
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Accuracy: {metrics.get('accuracy', 0):.2f}%")
    report.append(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.2f}%")
    report.append(f"Cohen's Kappa: {metrics.get('cohen_kappa', 0):.3f}")
    report.append(f"Matthews Correlation: {metrics.get('matthews_corrcoef', 0):.3f}")
    report.append(f"AUC-ROC (macro): {metrics.get('auc_macro', 0):.3f}")
    report.append(f"AUC-ROC (weighted): {metrics.get('auc_weighted', 0):.3f}")
    report.append("")
    
    # Medical Metrics
    report.append("MEDICAL METRICS")
    report.append("-" * 30)
    report.append(f"Mean Sensitivity: {metrics.get('mean_sensitivity', 0):.1f}%")
    report.append(f"Mean Specificity: {metrics.get('mean_specificity', 0):.1f}%")
    
    if 'dr_quadratic_kappa' in metrics:
        report.append(f"DR Quadratic Kappa: {metrics.get('dr_quadratic_kappa', 0):.3f}")
    report.append("")
    
    # Critical Conditions Performance
    if 'critical_alerts' in metrics:
        report.append("CRITICAL CONDITIONS ALERTS")
        report.append("-" * 30)
        
        for alert in metrics['critical_alerts']:
            report.append(f"\n{alert['class']} [ATTENTION REQUIRED]:")
            report.append(f"   Current Sensitivity: {alert['current_sensitivity']:.1f}%")
            report.append(f"   Required Sensitivity: {alert['required_sensitivity']:.0f}%")
            report.append(f"   Clinical Impact: {alert['reason']}")
    else:
        report.append("CRITICAL CONDITIONS PERFORMANCE")
        report.append("-" * 30)
        report.append("All critical conditions meet minimum sensitivity requirements ✓")
    report.append("")
    
    # Per-Class Performance
    report.append("PER-CLASS PERFORMANCE")
    report.append("-" * 30)
    report.append(f"{'Class':<20} {'Sens%':>8} {'Spec%':>8} {'PPV%':>8} {'NPV%':>8} {'AUC':>8}")
    report.append("-" * 68)
    
    for class_name in class_names:
        sens = metrics.get(f'{class_name}_sensitivity', 0)
        spec = metrics.get(f'{class_name}_specificity', 0)
        ppv = metrics.get(f'{class_name}_ppv', 0)
        npv = metrics.get(f'{class_name}_npv', 0)
        auc_score = metrics.get(f'{class_name}_auc', 0)
        
        report.append(
            f"{class_name:<20} {sens:8.1f} {spec:8.1f} {ppv:8.1f} {npv:8.1f} {auc_score:8.3f}"
        )
    report.append("")
    
    # Recommendations
    if include_recommendations:
        report.append("CLINICAL RECOMMENDATIONS")
        report.append("-" * 30)
        
        recommendations = []
        
        # Check overall performance
        if metrics.get('accuracy', 0) < 90:
            recommendations.append("Overall accuracy below 90% - additional training recommended")
        
        # Check sensitivity for critical conditions
        if 'critical_alerts' in metrics and metrics['critical_alerts']:
            recommendations.append("Critical conditions below required sensitivity - immediate attention needed")
        
        # Check balance
        if abs(metrics.get('mean_sensitivity', 0) - metrics.get('mean_specificity', 0)) > 10:
            recommendations.append("Significant imbalance between sensitivity and specificity")
        
        # General recommendations
        if not recommendations:
            recommendations = [
                "Model shows good overall performance",
                "External validation on independent dataset recommended",
                "Clinical trial with human expert comparison advised",
                "Monitor performance in production environment",
                "Regular retraining with new data suggested"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
    
    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    
    # Join report
    report_text = '\n'.join(report)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Clinical report saved to {output_path}")
    
    return report_text


def generate_training_report(
    training_history: Dict[str, Any],
    final_metrics: Dict[str, Any],
    config: Any,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive training report
    
    Args:
        training_history: Training history dictionary
        final_metrics: Final evaluation metrics
        config: Training configuration
        output_path: Path to save report
        
    Returns:
        Report dictionary
    """
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': config.model_type if hasattr(config, 'model_type') else 'unknown',
            'version': '1.0'
        },
        'dataset': {
            'path': str(config.dataset_path) if hasattr(config, 'dataset_path') else 'unknown',
            'num_classes': config.num_classes if hasattr(config, 'num_classes') else 0,
            'input_size': config.input_size if hasattr(config, 'input_size') else 224
        },
        'training': {
            'total_epochs': len(training_history.get('train_loss', [])),
            'batch_size': config.batch_size if hasattr(config, 'batch_size') else 0,
            'learning_rate': config.base_lr if hasattr(config, 'base_lr') else 0,
            'optimizer': config.optimizer if hasattr(config, 'optimizer') else 'unknown',
            'total_time_hours': sum(training_history.get('epoch_time', [])) / 3600
        },
        'performance': {
            'final_train_loss': training_history.get('train_loss', [0])[-1],
            'final_val_loss': training_history.get('val_loss', [0])[-1],
            'final_train_acc': training_history.get('train_acc', [0])[-1],
            'final_val_acc': training_history.get('val_acc', [0])[-1],
            'best_val_acc': max(training_history.get('val_acc', [0])),
            'best_epoch': np.argmax(training_history.get('val_acc', [0]))
        },
        'final_metrics': final_metrics,
        'optimizations': {
            'mixed_precision': config.use_amp if hasattr(config, 'use_amp') else False,
            'gradient_checkpointing': config.use_gradient_checkpointing if hasattr(config, 'use_gradient_checkpointing') else False,
            'sam_optimizer': config.use_sam if hasattr(config, 'use_sam') else False,
            'ema': config.use_ema if hasattr(config, 'use_ema') else False,
            'tta': config.use_tta if hasattr(config, 'use_tta') else False
        }
    }
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report saved to {output_path}")
    
    return report


def generate_model_card(
    model_info: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    training_details: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate model card following best practices
    
    Args:
        model_info: Model information
        performance_metrics: Performance metrics
        training_details: Training details
        output_path: Path to save model card
        
    Returns:
        Model card as markdown string
    """
    card = []
    
    # Header
    card.append(f"# {model_info.get('name', 'Model')} Model Card")
    card.append("")
    
    # Model Details
    card.append("## Model Details")
    card.append("")
    card.append(f"- **Model Type**: {model_info.get('architecture', 'Unknown')}")
    card.append(f"- **Version**: {model_info.get('version', '1.0')}")
    card.append(f"- **Training Date**: {model_info.get('date', datetime.now().strftime('%Y-%m-%d'))}")
    card.append(f"- **Framework**: PyTorch {model_info.get('pytorch_version', 'Unknown')}")
    card.append("")
    
    # Intended Use
    card.append("## Intended Use")
    card.append("")
    card.append("### Primary intended uses")
    card.append("- Screening tool for ophthalmological conditions")
    card.append("- Research purposes in medical imaging")
    card.append("- Educational demonstrations")
    card.append("")
    card.append("### Out-of-scope use cases")
    card.append("- Primary diagnostic tool without human oversight")
    card.append("- Clinical decision making without validation")
    card.append("- Use on populations not represented in training data")
    card.append("")
    
    # Training Data
    card.append("## Training Data")
    card.append("")
    card.append(f"- **Dataset**: {training_details.get('dataset_name', 'Unknown')}")
    card.append(f"- **Total Images**: {training_details.get('total_images', 0):,}")
    card.append(f"- **Number of Classes**: {training_details.get('num_classes', 0)}")
    card.append(f"- **Image Resolution**: {training_details.get('input_size', 224)}x{training_details.get('input_size', 224)}")
    card.append("")
    
    # Performance
    card.append("## Performance")
    card.append("")
    card.append("### Overall Metrics")
    card.append(f"- **Accuracy**: {performance_metrics.get('accuracy', 0):.2f}%")
    card.append(f"- **AUC-ROC**: {performance_metrics.get('auc_macro', 0):.3f}")
    card.append(f"- **Cohen's Kappa**: {performance_metrics.get('cohen_kappa', 0):.3f}")
    card.append("")
    
    # Limitations
    card.append("## Limitations")
    card.append("")
    card.append("- Model performance may vary on different populations")
    card.append("- Not validated for clinical use")
    card.append("- May exhibit bias based on training data distribution")
    card.append("- Performance on rare conditions may be limited")
    card.append("")
    
    # Ethical Considerations
    card.append("## Ethical Considerations")
    card.append("")
    card.append("- **Bias**: Model may exhibit demographic biases")
    card.append("- **Privacy**: Ensure patient data is properly de-identified")
    card.append("- **Clinical Use**: Requires validation and regulatory approval")
    card.append("- **Transparency**: Decision process should be explainable")
    card.append("")
    
    # Citation
    card.append("## Citation")
    card.append("")
    card.append("```bibtex")
    card.append("@software{retfound_implementation,")
    card.append("  title={RETFound Implementation for Ophthalmology},")
    card.append(f"  author={{CAASI Medical AI}},")
    card.append(f"  year={{{datetime.now().year}}},")
    card.append("  url={https://github.com/caasi/retfound}")
    card.append("}")
    card.append("```")
    
    # Join card
    card_text = '\n'.join(card)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(card_text)
        logger.info(f"Model card saved to {output_path}")
    
    return card_text


def create_latex_report(
    metrics: Dict[str, Any],
    figures: Dict[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Create LaTeX report for publication
    
    Args:
        metrics: Metrics dictionary
        figures: Dictionary of figure paths
        output_path: Path to save LaTeX file
    """
    latex = []
    
    # Document setup
    latex.append(r"\documentclass[11pt]{article}")
    latex.append(r"\usepackage{graphicx}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(r"\usepackage{float}")
    latex.append(r"\usepackage{amsmath}")
    latex.append(r"\begin{document}")
    latex.append("")
    
    # Title
    latex.append(r"\title{Medical AI Model Evaluation Report}")
    latex.append(r"\author{CAASI Medical AI}")
    latex.append(r"\date{\today}")
    latex.append(r"\maketitle")
    latex.append("")
    
    # Abstract
    latex.append(r"\begin{abstract}")
    latex.append(f"This report presents the evaluation results of a deep learning model "
                f"for ophthalmological image classification. The model achieved an overall "
                f"accuracy of {metrics.get('accuracy', 0):.2f}\% with an AUC-ROC of "
                f"{metrics.get('auc_macro', 0):.3f}.")
    latex.append(r"\end{abstract}")
    latex.append("")
    
    # Results section
    latex.append(r"\section{Results}")
    latex.append("")
    
    # Performance table
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Overall Performance Metrics}")
    latex.append(r"\begin{tabular}{lc}")
    latex.append(r"\toprule")
    latex.append(r"Metric & Value \\")
    latex.append(r"\midrule")
    latex.append(f"Accuracy & {metrics.get('accuracy', 0):.2f}\% \\\\")
    latex.append(f"Balanced Accuracy & {metrics.get('balanced_accuracy', 0):.2f}\% \\\\")
    latex.append(f"Cohen's Kappa & {metrics.get('cohen_kappa', 0):.3f} \\\\")
    latex.append(f"AUC-ROC (macro) & {metrics.get('auc_macro', 0):.3f} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    
    # Figures
    if 'confusion_matrix' in figures:
        latex.append(r"\begin{figure}[H]")
        latex.append(r"\centering")
        latex.append(f"\\includegraphics[width=0.8\\textwidth]{{{figures['confusion_matrix']}}}")
        latex.append(r"\caption{Confusion Matrix}")
        latex.append(r"\end{figure}")
    
    latex.append(r"\end{document}")
    
    # Save
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    logger.info(f"LaTeX report saved to {output_path}")


def export_metrics_to_csv(
    metrics: Dict[str, Any],
    class_names: List[str],
    output_path: Union[str, Path]
) -> None:
    """
    Export metrics to CSV format
    
    Args:
        metrics: Metrics dictionary
        class_names: Names of classes
        output_path: Path to save CSV file
    """
    rows = []
    
    # Overall metrics
    rows.append(['Metric', 'Value'])
    rows.append(['Accuracy', f"{metrics.get('accuracy', 0):.2f}"])
    rows.append(['Balanced Accuracy', f"{metrics.get('balanced_accuracy', 0):.2f}"])
    rows.append(['Cohen Kappa', f"{metrics.get('cohen_kappa', 0):.3f}"])
    rows.append(['AUC-ROC (macro)', f"{metrics.get('auc_macro', 0):.3f}"])
    rows.append([])  # Empty row
    
    # Per-class metrics
    rows.append(['Class', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC'])
    
    for class_name in class_names:
        row = [
            class_name,
            f"{metrics.get(f'{class_name}_sensitivity', 0):.2f}",
            f"{metrics.get(f'{class_name}_specificity', 0):.2f}",
            f"{metrics.get(f'{class_name}_ppv', 0):.2f}",
            f"{metrics.get(f'{class_name}_npv', 0):.2f}",
            f"{metrics.get(f'{class_name}_auc', 0):.3f}"
        ]
        rows.append(row)
    
    # Save
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    logger.info(f"Metrics exported to {output_path}")