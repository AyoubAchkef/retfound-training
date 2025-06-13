"""
Evaluate command for RETFound CLI.
Supports evaluation of models trained on v6.1 dataset with 28 classes.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from ...core.constants import DATASET_V61_CLASSES, CRITICAL_CONDITIONS
from ...evaluation import RETFoundEvaluator
from ...utils.logging import get_logger

logger = get_logger(__name__)


def add_evaluate_args(parser: argparse.ArgumentParser):
    """Add evaluate command arguments."""
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to dataset (overrides config)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--dataset-version',
        type=str,
        default='auto',
        choices=['auto', 'v4.0', 'v6.1'],
        help='Dataset version (auto-detect by default)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        choices=['fundus', 'oct', 'both'],
        help='Modality to evaluate (v6.1 only)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Use test-time augmentation'
    )
    parser.add_argument(
        '--temperature-scaling',
        action='store_true',
        help='Apply temperature scaling'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='Save predictions to file'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save visualization plots'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        default=True,
        help='Save detailed evaluation report'
    )
    parser.add_argument(
        '--clinical-report',
        action='store_true',
        help='Generate clinical-style report'
    )
    
    # Device options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index to use'
    )


def run_evaluate(args: argparse.Namespace) -> Dict:
    """
    Run evaluation command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 60)
    logger.info("RETFound Evaluation")
    logger.info("=" * 60)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Auto-detect dataset version if needed
    dataset_version = args.dataset_version
    if dataset_version == 'auto':
        dataset_version = _detect_dataset_version(args.checkpoint)
        logger.info(f"Auto-detected dataset version: {dataset_version}")
    
    # Validate modality argument
    if args.modality and dataset_version != 'v6.1':
        logger.warning("Modality filtering is only supported for v6.1 dataset")
        args.modality = None
    
    # Create evaluator
    evaluator = RETFoundEvaluator(
        checkpoint_path=args.checkpoint,
        device=device,
        save_dir=args.output_dir,
        dataset_version=dataset_version,
        modality=args.modality
    )
    
    # Log evaluation configuration
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset version: {dataset_version}")
    logger.info(f"Split: {args.split}")
    if args.modality:
        logger.info(f"Modality: {args.modality}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.tta:
        logger.info("Test-time augmentation: ENABLED")
    if args.temperature_scaling:
        logger.info("Temperature scaling: ENABLED")
    
    # Run evaluation
    try:
        results = evaluator.evaluate(
            split=args.split,
            save_predictions=args.save_predictions,
            save_plots=args.save_plots,
            save_report=args.save_report,
            tta=args.tta,
            temperature_scaling=args.temperature_scaling
        )
        
        # Generate clinical report if requested
        if args.clinical_report:
            _generate_clinical_report(evaluator, results, args)
        
        # Print summary
        _print_evaluation_summary(results, dataset_version)
        
        # Check critical conditions (v6.1 only)
        if dataset_version == 'v6.1':
            _check_critical_conditions(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def _detect_dataset_version(checkpoint_path: str) -> str:
    """Auto-detect dataset version from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check metadata
        if 'metadata' in checkpoint:
            if 'dataset_version' in checkpoint['metadata']:
                return checkpoint['metadata']['dataset_version']
        
        # Check config
        if 'config' in checkpoint:
            if 'dataset_version' in checkpoint['config'].get('data', {}):
                return checkpoint['config']['data']['dataset_version']
        
        # Check model
        if 'model_state_dict' in checkpoint:
            # Look for final layer size
            for key, value in checkpoint['model_state_dict'].items():
                if 'head.weight' in key or 'fc.weight' in key:
                    num_classes = value.shape[0]
                    if num_classes == 28:
                        return 'v6.1'
                    elif num_classes == 22:
                        return 'v4.0'
        
        # Default to v6.1
        logger.warning("Could not auto-detect dataset version, defaulting to v6.1")
        return 'v6.1'
        
    except Exception as e:
        logger.warning(f"Error detecting dataset version: {str(e)}, defaulting to v6.1")
        return 'v6.1'


def _print_evaluation_summary(results: Dict, dataset_version: str):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Overall metrics
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {results.get('balanced_accuracy', 0):.4f}")
    print(f"  Macro F1: {results.get('f1_macro', 0):.4f}")
    print(f"  Weighted F1: {results.get('f1_weighted', 0):.4f}")
    print(f"  Cohen's Kappa: {results.get('cohen_kappa', 0):.4f}")
    
    if 'quadratic_kappa' in results:
        print(f"  Quadratic Kappa (DR): {results['quadratic_kappa']:.4f}")
    
    if 'auc_macro' in results:
        print(f"  Macro AUC: {results['auc_macro']:.4f}")
    
    # Modality-specific metrics (v6.1)
    if dataset_version == 'v6.1':
        if 'fundus_accuracy' in results:
            print(f"\nFundus Performance:")
            print(f"  Accuracy: {results['fundus_accuracy']:.4f}")
            if 'fundus_correct_modality' in results:
                print(f"  Correct Modality: {results['fundus_correct_modality']:.4f}")
        
        if 'oct_accuracy' in results:
            print(f"\nOCT Performance:")
            print(f"  Accuracy: {results['oct_accuracy']:.4f}")
            if 'oct_correct_modality' in results:
                print(f"  Correct Modality: {results['oct_correct_modality']:.4f}")
    
    # Top-5 best performing classes
    print("\nTop 5 Best Performing Classes:")
    class_f1_scores = []
    for key, value in results.items():
        if key.startswith('f1_') and not key.endswith(('macro', 'weighted')):
            class_name = key.replace('f1_', '')
            class_f1_scores.append((class_name, value))
    
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    for i, (class_name, f1) in enumerate(class_f1_scores[:5], 1):
        print(f"  {i}. {class_name}: {f1:.4f}")
    
    # Bottom-5 worst performing classes
    print("\nBottom 5 Worst Performing Classes:")
    for i, (class_name, f1) in enumerate(class_f1_scores[-5:], 1):
        print(f"  {i}. {class_name}: {f1:.4f}")


def _check_critical_conditions(results: Dict):
    """Check performance on critical conditions."""
    print("\n" + "=" * 60)
    print("CRITICAL CONDITIONS ANALYSIS")
    print("=" * 60)
    
    all_pass = True
    critical_results = []
    
    for condition, info in CRITICAL_CONDITIONS.items():
        sens_key = f'critical_{condition}_sensitivity'
        if sens_key in results:
            sensitivity = results[sens_key]
            threshold = info['min_sensitivity']
            meets = results.get(f'critical_{condition}_meets_threshold', False)
            
            critical_results.append({
                'condition': condition,
                'sensitivity': sensitivity,
                'threshold': threshold,
                'meets': meets,
                'severity': info['severity'],
                'urgency': info['urgency']
            })
            
            if not meets:
                all_pass = False
    
    # Sort by severity
    critical_results.sort(key=lambda x: (
        ['low', 'moderate', 'high', 'critical'].index(x['severity']),
        x['sensitivity']
    ), reverse=True)
    
    # Print results
    for result in critical_results:
        status = "✓ PASS" if result['meets'] else "✗ FAIL"
        print(f"\n{result['condition']}:")
        print(f"  Sensitivity: {result['sensitivity']:.3f} (threshold: {result['threshold']:.2f}) {status}")
        print(f"  Severity: {result['severity']}")
        print(f"  Clinical urgency: {result['urgency']}")
    
    # Overall summary
    print("\n" + "-" * 60)
    if all_pass:
        print("✓ All critical conditions meet sensitivity thresholds!")
    else:
        print("✗ Some critical conditions are below threshold!")
        print("⚠️  Model may miss critical cases requiring urgent care")
    
    # Average critical sensitivity
    if 'critical_avg_sensitivity' in results:
        print(f"\nAverage Critical Sensitivity: {results['critical_avg_sensitivity']:.4f}")
    if 'critical_min_sensitivity' in results:
        print(f"Minimum Critical Sensitivity: {results['critical_min_sensitivity']:.4f}")


def _generate_clinical_report(evaluator: RETFoundEvaluator, results: Dict, args: argparse.Namespace):
    """Generate clinical-style evaluation report."""
    report_path = Path(evaluator.save_dir) / "clinical_evaluation_report.md"
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Clinical Evaluation Report - RETFound Model\n\n")
        f.write(f"**Model**: {Path(args.checkpoint).name}\n")
        f.write(f"**Dataset Version**: {evaluator.dataset_version}\n")
        f.write(f"**Evaluation Date**: {torch.datetime.datetime.now()}\n")
        if args.modality:
            f.write(f"**Modality**: {args.modality}\n")
        f.write("\n---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Overall Accuracy**: {results.get('accuracy', 0):.1%}\n")
        f.write(f"- **Balanced Accuracy**: {results.get('balanced_accuracy', 0):.1%}\n")
        f.write(f"- **Cohen's Kappa**: {results.get('cohen_kappa', 0):.3f}\n")
        
        if evaluator.dataset_version == 'v6.1':
            # Critical conditions summary
            critical_pass = all(
                results.get(f'critical_{cond}_meets_threshold', False)
                for cond in CRITICAL_CONDITIONS
                if f'critical_{cond}_sensitivity' in results
            )
            
            if critical_pass:
                f.write("- **Critical Conditions**: ✓ All meet clinical thresholds\n")
            else:
                f.write("- **Critical Conditions**: ⚠️ Some below clinical thresholds\n")
        
        f.write("\n")
        
        # Clinical Performance by Pathology
        f.write("## Clinical Performance by Pathology\n\n")
        
        # Group classes by clinical category
        clinical_categories = {
            'Diabetic Retinopathy': ['DR_Mild', 'DR_Moderate', 'DR_Severe', 'DR_Proliferative'],
            'Age-Related Macular Degeneration': ['CNV_Wet_AMD', 'CNV_OCT', 'Dry_AMD', 'Drusen'],
            'Vascular Conditions': ['RVO', 'RVO_OCT', 'RAO', 'RAO_OCT', 'Hypertensive_Retinopathy'],
            'Glaucoma': ['Glaucoma_Suspect', 'Glaucoma_Positive', 'Glaucoma_OCT'],
            'Other Retinal Conditions': ['CSR', 'ERM', 'Retinal_Detachment', 'Myopia_Degenerative']
        }
        
        for category, conditions in clinical_categories.items():
            f.write(f"### {category}\n\n")
            
            category_results = []
            for condition in conditions:
                if f'recall_{condition}' in results:
                    category_results.append({
                        'condition': condition,
                        'sensitivity': results[f'recall_{condition}'],
                        'specificity': results.get(f'specificity_{condition}', 0),
                        'f1': results.get(f'f1_{condition}', 0)
                    })
            
            if category_results:
                f.write("| Condition | Sensitivity | Specificity | F1-Score |\n")
                f.write("|-----------|-------------|-------------|----------|\n")
                
                for res in category_results:
                    f.write(f"| {res['condition']} | {res['sensitivity']:.1%} | "
                           f"{res['specificity']:.1%} | {res['f1']:.3f} |\n")
                
                # Category average
                avg_sens = sum(r['sensitivity'] for r in category_results) / len(category_results)
                f.write(f"\n**Category Average Sensitivity**: {avg_sens:.1%}\n\n")
            else:
                f.write("*No data available for this category*\n\n")
        
        # Clinical Recommendations
        f.write("## Clinical Recommendations\n\n")
        
        if evaluator.dataset_version == 'v6.1':
            # Check critical conditions
            failing_critical = []
            for condition in CRITICAL_CONDITIONS:
                if f'critical_{condition}_sensitivity' in results:
                    if not results.get(f'critical_{condition}_meets_threshold', False):
                        failing_critical.append(condition)
            
            if failing_critical:
                f.write("### ⚠️ Critical Conditions Requiring Attention\n\n")
                f.write("The following conditions fall below clinical safety thresholds:\n\n")
                for condition in failing_critical:
                    sensitivity = results.get(f'critical_{condition}_sensitivity', 0)
                    threshold = CRITICAL_CONDITIONS[condition]['min_sensitivity']
                    f.write(f"- **{condition}**: {sensitivity:.1%} "
                           f"(threshold: {threshold:.0%})\n")
                f.write("\n**Recommendation**: Additional training or model refinement "
                       "needed for these critical conditions.\n\n")
            else:
                f.write("### ✓ All Critical Conditions Meet Thresholds\n\n")
                f.write("The model meets clinical safety thresholds for all critical conditions.\n\n")
        
        # Limitations
        f.write("## Limitations and Disclaimers\n\n")
        f.write("1. This model is for research and screening purposes only\n")
        f.write("2. Clinical diagnosis should always be performed by qualified ophthalmologists\n")
        f.write("3. Performance may vary on different populations or imaging equipment\n")
        f.write("4. Regular revalidation is recommended as new data becomes available\n")
        
        # Appendix
        f.write("\n## Appendix: Full Classification Report\n\n")
        f.write("```\n")
        f.write(evaluator.metrics.get_classification_report())
        f.write("```\n")
    
    logger.info(f"Clinical report saved to {report_path}")


def add_subparser(subparsers):
    """Add evaluate subcommand to parser"""
    parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate RETFound model',
        description='Evaluate trained RETFound model on test data'
    )
    
    add_evaluate_args(parser)
    parser.set_defaults(func=run_evaluate)
    
    return parser
