"""
Evaluate Command
================

CLI command for evaluating trained RETFound models.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch
import numpy as np

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.data.datamodule import RETFoundDataModule
from retfound.metrics.medical import OphthalmologyMetrics
from retfound.utils.logging import setup_logging
from retfound.utils.device import get_device
from retfound.export.inference import TTAWrapper

logger = logging.getLogger(__name__)


def add_evaluate_args(parser):
    """Add evaluation-specific arguments to parser"""
    parser.add_argument('checkpoint', type=str,
                       help='Path to model checkpoint to evaluate')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if different from checkpoint)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Override dataset path')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Which data split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save evaluation results')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--tta-augmentations', type=int, default=5,
                       help='Number of TTA augmentations')
    parser.add_argument('--use-ema', action='store_true',
                       help='Use EMA model if available')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save all predictions to file')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed clinical report')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')


def load_model_from_checkpoint(checkpoint_path: Path, config: RETFoundConfig, 
                               use_ema: bool = False) -> torch.nn.Module:
    """Load model from checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = create_model(config)
    
    # Load weights
    if use_ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        logger.info("Loading EMA model weights")
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        logger.info("Loading regular model weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load temperature scaling if available
    temperature = checkpoint.get('temperature', None)
    if temperature is not None:
        logger.info(f"Temperature scaling value: {temperature}")
    
    return model, checkpoint.get('metrics', {}), temperature


def evaluate_model(model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader,
                   config: RETFoundConfig,
                   device: torch.device,
                   use_tta: bool = False,
                   tta_augmentations: int = 5) -> Dict[str, Any]:
    """Evaluate model on dataset"""
    model.eval()
    model = model.to(device)
    
    # Initialize metrics
    metrics = OphthalmologyMetrics(
        num_classes=config.num_classes,
        class_names=getattr(dataloader.dataset, 'class_names', None)
    )
    
    # TTA wrapper if needed
    if use_tta:
        logger.info(f"Using test-time augmentation with {tta_augmentations} augmentations")
        tta_config = config.copy()
        tta_config.tta_augmentations = tta_augmentations
        tta_wrapper = TTAWrapper(model, tta_config, device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Evaluation loop
    logger.info(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_tta:
                # TTA prediction
                batch_probs = []
                for i in range(images.size(0)):
                    img = images[i:i+1]
                    probs = tta_wrapper.predict(img)
                    batch_probs.append(probs)
                outputs = torch.cat(batch_probs, dim=0)
            else:
                # Regular prediction
                outputs = model(images)
            
            # Update metrics
            metrics.update(outputs, labels)
            
            # Store predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Compute final metrics
    final_metrics = metrics.compute_metrics()
    
    # Add raw predictions to results
    results = {
        'metrics': final_metrics,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities),
        'class_names': metrics.class_names
    }
    
    return results, metrics


def generate_clinical_report(results: Dict[str, Any], config: RETFoundConfig,
                           checkpoint_info: Dict[str, Any]) -> str:
    """Generate comprehensive clinical evaluation report"""
    report = []
    report.append("="*70)
    report.append("CLINICAL EVALUATION REPORT - RETFOUND")
    report.append("="*70)
    
    from datetime import datetime
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Model: RETFound ({config.model_type})")
    report.append(f"Checkpoint: {checkpoint_info.get('epoch', 'Unknown')} epochs")
    report.append(f"Dataset: {config.dataset_path}")
    report.append("")
    
    metrics = results['metrics']
    
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Accuracy: {metrics.get('accuracy', 0):.2f}%")
    report.append(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.2f}%")
    report.append(f"Cohen's Kappa: {metrics.get('cohen_kappa', 0):.3f}")
    report.append(f"Matthews Correlation: {metrics.get('mcc', 0):.3f}")
    report.append(f"AUC-ROC (macro): {metrics.get('auc_macro', 0):.3f}")
    report.append(f"AUC-ROC (weighted): {metrics.get('auc_weighted', 0):.3f}")
    report.append("")
    
    report.append("MEDICAL METRICS")
    report.append("-" * 30)
    report.append(f"Mean Sensitivity: {metrics.get('mean_sensitivity', 0):.1f}%")
    report.append(f"Mean Specificity: {metrics.get('mean_specificity', 0):.1f}%")
    
    if 'dr_quadratic_kappa' in metrics:
        report.append(f"DR Quadratic Kappa: {metrics.get('dr_quadratic_kappa', 0):.3f}")
    
    # Per-class performance
    report.append("")
    report.append("PER-CLASS PERFORMANCE")
    report.append("-" * 30)
    
    class_names = results.get('class_names', [])
    for i, class_name in enumerate(class_names[:config.num_classes]):
        sens = metrics.get(f'{class_name}_sensitivity', 0)
        spec = metrics.get(f'{class_name}_specificity', 0)
        ppv = metrics.get(f'{class_name}_ppv', 0)
        npv = metrics.get(f'{class_name}_npv', 0)
        f1 = metrics.get(f'{class_name}_f1', 0)
        auc = metrics.get(f'{class_name}_auc', 0)
        
        report.append(f"\n{class_name}:")
        report.append(f"   Sensitivity: {sens:.1f}%")
        report.append(f"   Specificity: {spec:.1f}%")
        report.append(f"   PPV: {ppv:.1f}%")
        report.append(f"   NPV: {npv:.1f}%")
        report.append(f"   F1-Score: {f1:.3f}")
        report.append(f"   AUC: {auc:.3f}")
    
    # Critical conditions check
    if 'critical_alerts' in metrics:
        report.append("")
        report.append("CRITICAL CONDITIONS ALERTS")
        report.append("-" * 30)
        for alert in metrics['critical_alerts']:
            report.append(f"⚠️ {alert['class']}: {alert['current_sensitivity']:.1f}% < {alert['required_sensitivity']:.0f}%")
            report.append(f"   Reason: {alert['reason']}")
    
    # Confusion matrix summary
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results['targets'], results['predictions'])
    
    report.append("")
    report.append("CONFUSION MATRIX SUMMARY")
    report.append("-" * 30)
    report.append("Top misclassifications:")
    
    # Find top 5 misclassifications
    misclass = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], class_names[i], class_names[j]))
    
    misclass.sort(reverse=True)
    for count, true_class, pred_class in misclass[:5]:
        report.append(f"   {true_class} → {pred_class}: {count} cases")
    
    report.append("")
    report.append("="*70)
    
    return "\n".join(report)


def run_evaluate(args) -> int:
    """Main evaluation function"""
    try:
        # Setup logging
        setup_logging()
        
        # Load checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load configuration
        if args.config:
            config = RETFoundConfig.load(Path(args.config))
            logger.info(f"Configuration loaded from {args.config}")
        elif 'config' in checkpoint:
            # Load config from checkpoint
            config = RETFoundConfig(**checkpoint['config'])
            logger.info("Configuration loaded from checkpoint")
        else:
            raise ValueError("No configuration found. Please provide --config")
        
        # Override settings
        if args.data_path:
            config.dataset_path = Path(args.data_path)
        if args.batch_size:
            config.batch_size = args.batch_size
        
        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = checkpoint_path.parent / 'evaluation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Load model
        model, checkpoint_metrics, temperature = load_model_from_checkpoint(
            checkpoint_path, config, use_ema=args.use_ema
        )
        
        # Setup data
        logger.info("Setting up data module...")
        data_module = RETFoundDataModule(config)
        data_module.setup()
        
        # Get appropriate dataloader
        if args.split == 'train':
            dataloader = data_module.train_dataloader()
        elif args.split == 'val':
            dataloader = data_module.val_dataloader()
        else:  # test
            dataloader = data_module.test_dataloader()
        
        logger.info(f"Evaluating on {args.split} split with {len(dataloader.dataset)} samples")
        
        # Evaluate
        results, metrics_obj = evaluate_model(
            model, dataloader, config, device,
            use_tta=args.use_tta,
            tta_augmentations=args.tta_augmentations
        )
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Dataset: {args.split}")
        logger.info(f"Samples: {len(results['targets'])}")
        logger.info(f"Accuracy: {results['metrics']['accuracy']:.2f}%")
        logger.info(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.2f}%")
        logger.info(f"Cohen's Kappa: {results['metrics']['cohen_kappa']:.3f}")
        logger.info(f"AUC-ROC (macro): {results['metrics']['auc_macro']:.3f}")
        
        # Save metrics
        metrics_path = output_dir / f'metrics_{args.split}.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy types for JSON
            json_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in results['metrics'].items()
            }
            json.dump(json_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_path = output_dir / f'predictions_{args.split}.npz'
            np.savez(
                predictions_path,
                predictions=results['predictions'],
                targets=results['targets'],
                probabilities=results['probabilities'],
                class_names=results['class_names']
            )
            logger.info(f"Predictions saved to {predictions_path}")
        
        # Generate plots
        if not args.no_plots:
            logger.info("Generating plots...")
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Confusion matrix
            cm_path = plots_dir / f'confusion_matrix_{args.split}.png'
            metrics_obj.plot_confusion_matrix(save_path=cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")
            
            # ROC curves
            roc_path = plots_dir / f'roc_curves_{args.split}.png'
            metrics_obj.plot_roc_curves(save_path=roc_path)
            logger.info(f"ROC curves saved to {roc_path}")
            
            # Calibration curves
            cal_path = plots_dir / f'calibration_curves_{args.split}.png'
            metrics_obj.plot_calibration_curves(save_path=cal_path)
            logger.info(f"Calibration curves saved to {cal_path}")
        
        # Generate clinical report
        if args.generate_report:
            logger.info("Generating clinical report...")
            report = generate_clinical_report(
                results, config, 
                {'epoch': checkpoint.get('epoch', 'Unknown')}
            )
            
            report_path = output_dir / f'clinical_report_{args.split}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Clinical report saved to {report_path}")
            
            # Also print to console
            print("\n" + report)
        
        logger.info(f"\nAll evaluation results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        logger.exception("Full traceback:")
        return 1
