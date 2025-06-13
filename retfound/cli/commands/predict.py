"""
Predict command for RETFound CLI.
Supports prediction with models trained on v6.1 dataset (28 classes).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch
from PIL import Image
import pandas as pd

from ...core.constants import DATASET_V61_CLASSES, CRITICAL_CONDITIONS
from ...export.inference import RETFoundPredictor
from ...utils.logging import get_logger

logger = get_logger(__name__)


def add_predict_args(parser: argparse.ArgumentParser):
    """Add predict command arguments."""
    parser.add_argument(
        'input',
        type=str,
        help='Input image path, directory, or CSV file with image paths'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to exported model (.onnx, .pt, .trt) or checkpoint (.pth)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        help='Path to model metadata JSON (auto-detected if not provided)'
    )
    
    # Prediction options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for prediction'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--check-critical',
        action='store_true',
        default=True,
        help='Check for critical conditions'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for predictions (CSV or JSON)'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='auto',
        choices=['auto', 'csv', 'json'],
        help='Output format'
    )
    parser.add_argument(
        '--save-probabilities',
        action='store_true',
        help='Save all class probabilities'
    )
    parser.add_argument(
        '--clinical-report',
        action='store_true',
        help='Generate clinical-style report for predictions'
    )
    
    # Device options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'tensorrt'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index to use'
    )
    
    # Display options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed predictions'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output except errors'
    )


def run_predict(args: argparse.Namespace) -> List[Dict]:
    """
    Run predict command.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of prediction results
    """
    if not args.quiet:
        logger.info("=" * 60)
        logger.info("RETFound Prediction")
        logger.info("=" * 60)
    
    # Setup device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = f'cuda:{args.gpu}'
            if not args.quiet:
                logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
        else:
            device = 'cpu'
            logger.warning("CUDA not available, falling back to CPU")
    else:
        device = args.device
        if not args.quiet:
            logger.info(f"Using device: {device}")
    
    # Get input images
    image_paths = _get_input_images(args.input)
    if not args.quiet:
        logger.info(f"Found {len(image_paths)} images to process")
    
    # Create predictor
    try:
        predictor = RETFoundPredictor(
            model_path=args.model,
            metadata_path=args.metadata,
            device=device,
            batch_size=args.batch_size
        )
        
        if not args.quiet:
            logger.info(f"Loaded model: {args.model}")
            logger.info(f"Dataset version: {predictor.dataset_version}")
            logger.info(f"Number of classes: {predictor.num_classes}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Run predictions
    try:
        results = predictor.predict_batch(
            image_paths=image_paths,
            save_results=False  # We'll handle saving ourselves
        )
        
        # Filter by threshold if specified
        if args.threshold > 0:
            results = _filter_by_threshold(results, args.threshold)
        
        # Display results
        if not args.quiet:
            _display_results(results, args)
        
        # Check for critical conditions
        if args.check_critical and predictor.dataset_version == 'v6.1':
            critical_summary = _analyze_critical_conditions(results)
            if not args.quiet and critical_summary['total_critical'] > 0:
                _display_critical_summary(critical_summary)
        
        # Save results
        if args.output:
            _save_results(results, args)
        
        # Generate clinical report if requested
        if args.clinical_report:
            _generate_clinical_report(results, predictor, args)
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def _get_input_images(input_path: str) -> List[Path]:
    """Get list of image paths from input."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single image
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return [input_path]
        
        # CSV file with image paths
        elif input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
            # Look for image path column
            path_columns = ['path', 'image_path', 'filepath', 'filename', 'image']
            for col in path_columns:
                if col in df.columns:
                    return [Path(p) for p in df[col].tolist()]
            
            # If no path column found, assume first column
            return [Path(p) for p in df.iloc[:, 0].tolist()]
        
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
    
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        
        return sorted(image_paths)
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def _filter_by_threshold(results: List[Dict], threshold: float) -> List[Dict]:
    """Filter results by confidence threshold."""
    filtered = []
    
    for result in results:
        if result['confidence'] >= threshold:
            filtered.append(result)
        else:
            # Still include but mark as below threshold
            result['below_threshold'] = True
            filtered.append(result)
    
    return filtered


def _display_results(results: List[Dict], args: argparse.Namespace):
    """Display prediction results."""
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results):
        # Get image name
        image_name = result['metadata'].get('filename', f'Image {i+1}')
        
        print(f"\n{image_name}:")
        print("-" * 40)
        
        # Main prediction
        if result.get('below_threshold', False):
            print(f"  Prediction: {result['predicted_class']} "
                  f"(confidence: {result['confidence']:.1%}) ⚠️ Below threshold")
        else:
            print(f"  Prediction: {result['predicted_class']} "
                  f"(confidence: {result['confidence']:.1%})")
        
        # Show modality for v6.1
        if 'modality' in result:
            print(f"  Modality: {result['modality'].upper()}")
        
        # Top-k predictions if verbose
        if args.verbose and 'top5_predictions' in result:
            print(f"\n  Top {args.top_k} predictions:")
            for j, pred in enumerate(result['top5_predictions'][:args.top_k], 1):
                print(f"    {j}. {pred['class']}: {pred['probability']:.1%}")
        
        # Critical conditions
        if 'critical_analysis' in result:
            analysis = result['critical_analysis']
            if analysis['is_critical']:
                print("\n  ⚠️ CRITICAL CONDITION DETECTED!")
                for cond in analysis['critical_conditions_detected']:
                    print(f"    - {cond['condition']} (severity: {cond['severity']})")
                print(f"    Recommendation: {analysis['recommendation']}")


def _analyze_critical_conditions(results: List[Dict]) -> Dict:
    """Analyze results for critical conditions."""
    summary = {
        'total_images': len(results),
        'total_critical': 0,
        'critical_conditions': {},
        'critical_images': []
    }
    
    for result in results:
        if 'critical_analysis' in result and result['critical_analysis']['is_critical']:
            summary['total_critical'] += 1
            summary['critical_images'].append(result['metadata'].get('filename', 'Unknown'))
            
            # Count by condition
            for cond in result['critical_analysis']['critical_conditions_detected']:
                condition_name = cond['condition']
                if condition_name not in summary['critical_conditions']:
                    summary['critical_conditions'][condition_name] = 0
                summary['critical_conditions'][condition_name] += 1
    
    return summary


def _display_critical_summary(summary: Dict):
    """Display critical conditions summary."""
    print("\n" + "=" * 80)
    print("CRITICAL CONDITIONS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal images analyzed: {summary['total_images']}")
    print(f"Images with critical conditions: {summary['total_critical']} "
          f"({summary['total_critical']/summary['total_images']*100:.1f}%)")
    
    if summary['critical_conditions']:
        print("\nBreakdown by condition:")
        for condition, count in sorted(summary['critical_conditions'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  - {condition}: {count}")
    
    print("\n⚠️  These cases require urgent ophthalmological evaluation!")


def _save_results(results: List[Dict], args: argparse.Namespace):
    """Save prediction results to file."""
    output_path = Path(args.output)
    
    # Auto-detect format from extension
    if args.output_format == 'auto':
        if output_path.suffix.lower() == '.csv':
            output_format = 'csv'
        elif output_path.suffix.lower() == '.json':
            output_format = 'json'
        else:
            output_format = 'csv'  # Default
    else:
        output_format = args.output_format
    
    if output_format == 'csv':
        _save_results_csv(results, output_path, args)
    else:
        _save_results_json(results, output_path)
    
    logger.info(f"Results saved to {output_path}")


def _save_results_csv(results: List[Dict], output_path: Path, args: argparse.Namespace):
    """Save results as CSV."""
    rows = []
    
    for result in results:
        row = {
            'image_path': result['metadata'].get('path', ''),
            'filename': result['metadata'].get('filename', ''),
            'predicted_class': result['predicted_class'],
            'predicted_class_idx': result['predicted_class_idx'],
            'confidence': result['confidence']
        }
        
        # Add modality if present
        if 'modality' in result:
            row['modality'] = result['modality']
        
        # Add critical status
        if 'critical_analysis' in result:
            row['is_critical'] = result['critical_analysis']['is_critical']
            if result['critical_analysis']['is_critical']:
                row['critical_condition'] = result['critical_analysis']['critical_conditions_detected'][0]['condition']
                row['recommendation'] = result['critical_analysis']['recommendation']
        
        # Add top-k predictions
        if 'top5_predictions' in result:
            for i, pred in enumerate(result['top5_predictions'][:args.top_k], 1):
                row[f'top{i}_class'] = pred['class']
                row[f'top{i}_probability'] = pred['probability']
        
        # Add all probabilities if requested
        if args.save_probabilities and 'probabilities' in result:
            for class_name, prob in result['probabilities'].items():
                row[f'prob_{class_name}'] = prob
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def _save_results_json(results: List[Dict], output_path: Path):
    """Save results as JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def _generate_clinical_report(results: List[Dict], predictor: RETFoundPredictor, args: argparse.Namespace):
    """Generate clinical-style report for predictions."""
    report_path = Path(args.output).parent / "clinical_prediction_report.md" if args.output else Path("clinical_prediction_report.md")
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Clinical Prediction Report - RETFound\n\n")
        f.write(f"**Model**: {Path(args.model).name}\n")
        f.write(f"**Dataset Version**: {predictor.dataset_version}\n")
        f.write(f"**Date**: {torch.datetime.datetime.now()}\n")
        f.write(f"**Total Images**: {len(results)}\n")
        f.write("\n---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        
        # Count by predicted class
        class_counts = {}
        modality_counts = {'fundus': 0, 'oct': 0}
        critical_count = 0
        
        for result in results:
            # Class counts
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            # Modality counts
            if 'modality' in result:
                modality_counts[result['modality']] += 1
            
            # Critical conditions
            if result.get('critical_analysis', {}).get('is_critical', False):
                critical_count += 1
        
        # Write summary statistics
        f.write(f"- **Normal cases**: {class_counts.get('Normal_Fundus', 0) + class_counts.get('Normal_OCT', 0)}\n")
        f.write(f"- **Pathological cases**: {len(results) - class_counts.get('Normal_Fundus', 0) - class_counts.get('Normal_OCT', 0)}\n")
        f.write(f"- **Critical conditions**: {critical_count}\n")
        
        if predictor.dataset_version == 'v6.1':
            f.write(f"\n**By Modality**:\n")
            f.write(f"- Fundus images: {modality_counts['fundus']}\n")
            f.write(f"- OCT images: {modality_counts['oct']}\n")
        
        # Distribution of conditions
        f.write("\n## Distribution of Detected Conditions\n\n")
        
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        f.write("| Condition | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        
        for class_name, count in sorted_classes:
            percentage = count / len(results) * 100
            f.write(f"| {class_name} | {count} | {percentage:.1f}% |\n")
        
        # Critical conditions detail
        if critical_count > 0:
            f.write("\n## Critical Conditions Requiring Urgent Attention\n\n")
            
            critical_by_type = {}
            for result in results:
                if result.get('critical_analysis', {}).get('is_critical', False):
                    for cond in result['critical_analysis']['critical_conditions_detected']:
                        condition_name = cond['condition']
                        if condition_name not in critical_by_type:
                            critical_by_type[condition_name] = []
                        critical_by_type[condition_name].append({
                            'filename': result['metadata'].get('filename', 'Unknown'),
                            'confidence': result['confidence'],
                            'severity': cond['severity'],
                            'urgency': cond['urgency']
                        })
            
            for condition, cases in critical_by_type.items():
                f.write(f"### {condition} ({len(cases)} cases)\n\n")
                f.write(f"**Severity**: {cases[0]['severity']}\n")
                f.write(f"**Clinical urgency**: {cases[0]['urgency']}\n\n")
                
                f.write("Affected images:\n")
                for case in cases[:10]:  # Show max 10
                    f.write(f"- {case['filename']} (confidence: {case['confidence']:.1%})\n")
                
                if len(cases) > 10:
                    f.write(f"- ... and {len(cases) - 10} more\n")
                f.write("\n")
        
        # Recommendations
        f.write("## Clinical Recommendations\n\n")
        
        if critical_count > 0:
            f.write(f"⚠️ **{critical_count} images show critical conditions requiring urgent evaluation.**\n\n")
            f.write("Recommended actions:\n")
            f.write("1. Prioritize review of critical cases\n")
            f.write("2. Schedule urgent ophthalmological consultations\n")
            f.write("3. Consider immediate referral for sight-threatening conditions\n\n")
        
        f.write("General recommendations:\n")
        f.write("- All predictions should be reviewed by qualified ophthalmologists\n")
        f.write("- Consider patient history and clinical context\n")
        f.write("- Use predictions as screening aids, not definitive diagnoses\n")
        
        # Detailed results (optional)
        if args.verbose:
            f.write("\n## Detailed Results\n\n")
            
            for result in results[:50]:  # Limit to first 50
                f.write(f"### {result['metadata'].get('filename', 'Unknown')}\n\n")
                f.write(f"- **Prediction**: {result['predicted_class']}\n")
                f.write(f"- **Confidence**: {result['confidence']:.1%}\n")
                
                if 'modality' in result:
                    f.write(f"- **Modality**: {result['modality']}\n")
                
                if result.get('critical_analysis', {}).get('is_critical', False):
                    f.write(f"- **Critical**: YES - {result['critical_analysis']['recommendation']}\n")
                
                f.write("\n")
        
        # Footer
        f.write("\n---\n\n")
        f.write("*This report is generated by an AI model and should not be used as the sole basis for clinical decisions.*\n")
    
    logger.info(f"Clinical report saved to {report_path}")


def add_subparser(subparsers):
    """Add predict subcommand to parser"""
    parser = subparsers.add_parser(
        'predict',
        help='Run predictions with RETFound model',
        description='Run inference on images using trained RETFound model'
    )
    
    add_predict_args(parser)
    parser.set_defaults(func=run_predict)
    
    return parser
