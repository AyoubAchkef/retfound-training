"""
Predict Command
===============

CLI command for making predictions with trained RETFound models.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import time

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.data.transforms import create_eval_transform
from retfound.export.inference import TTAWrapper
from retfound.utils.logging import setup_logging
from retfound.utils.device import get_device

logger = logging.getLogger(__name__)


def add_predict_args(parser):
    """Add prediction-specific arguments to parser"""
    parser.add_argument('input', type=str,
                       help='Input image, directory, or CSV file with image paths')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (JSON or CSV)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if different from checkpoint)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for prediction')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Return top-k predictions')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Confidence threshold for predictions')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--tta-augmentations', type=int, default=5,
                       help='Number of TTA augmentations')
    parser.add_argument('--use-ema', action='store_true',
                       help='Use EMA model if available')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for prediction')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'json', 'csv', 'detailed'],
                       help='Output format')
    parser.add_argument('--include-probabilities', action='store_true',
                       help='Include all class probabilities')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of predictions')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively search directories for images')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='Image file extensions to process')


class RETFoundPredictor:
    """Predictor class for RETFound models"""
    
    def __init__(self, checkpoint_path: str, config: Optional[RETFoundConfig] = None,
                 device: str = 'cuda', use_ema: bool = False):
        self.device = get_device(device)
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load or create config
        if config is None:
            if 'config' in self.checkpoint:
                self.config = RETFoundConfig(**self.checkpoint['config'])
            else:
                raise ValueError("No configuration found in checkpoint")
        else:
            self.config = config
        
        # Create and load model
        self.model = create_model(self.config)
        
        if use_ema and 'ema_state_dict' in self.checkpoint:
            logger.info("Loading EMA model weights")
            self.model.load_state_dict(self.checkpoint['ema_state_dict'])
        else:
            logger.info("Loading regular model weights")
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get class names
        self.class_names = self.checkpoint.get('class_names', 
                                               [f'Class_{i}' for i in range(self.config.num_classes)])
        
        # Temperature scaling
        self.temperature = self.checkpoint.get('temperature', 1.0)
        if self.temperature != 1.0:
            logger.info(f"Using temperature scaling: {self.temperature}")
        
        # Create transform
        self.transform = create_eval_transform(self.config)
        
        # TTA wrapper
        self.tta_wrapper = None
    
    def enable_tta(self, num_augmentations: int = 5):
        """Enable test-time augmentation"""
        tta_config = self.config.copy()
        tta_config.tta_augmentations = num_augmentations
        self.tta_wrapper = TTAWrapper(self.model, tta_config, self.device)
        logger.info(f"TTA enabled with {num_augmentations} augmentations")
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess single image"""
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if hasattr(self.transform, '__call__'):
            # Albumentations
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            # torchvision
            image_tensor = self.transform(image)
        
        return image_tensor
    
    def predict_single(self, image_path: Union[str, Path], top_k: int = 5) -> Dict[str, Any]:
        """Predict for single image"""
        image_path = Path(image_path)
        
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        start_time = time.time()
        
        with torch.no_grad():
            if self.tta_wrapper:
                # TTA prediction
                probs = self.tta_wrapper.predict(image_batch)
                logits = torch.log(probs / (1 - probs + 1e-8))  # Inverse softmax
            else:
                # Regular prediction
                logits = self.model(image_batch)
                logits = logits / self.temperature
                probs = torch.softmax(logits, dim=1)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get top predictions
        probs_cpu = probs[0].cpu()
        top_probs, top_indices = torch.topk(probs_cpu, k=min(top_k, len(self.class_names)))
        
        # Build result
        result = {
            'image': str(image_path),
            'predicted_class': top_indices[0].item(),
            'predicted_label': self.class_names[top_indices[0].item()],
            'confidence': top_probs[0].item(),
            'top_k_predictions': [],
            'inference_time_ms': inference_time
        }
        
        for i in range(len(top_indices)):
            result['top_k_predictions'].append({
                'class_id': top_indices[i].item(),
                'class_name': self.class_names[top_indices[i].item()],
                'probability': top_probs[i].item()
            })
        
        return result
    
    def predict_batch(self, image_paths: List[Path], batch_size: int = 32,
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """Predict for batch of images"""
        results = []
        
        # Process in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Processing batches")
        
        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    results.append({
                        'image': str(path),
                        'error': str(e)
                    })
            
            if not batch_tensors:
                continue
            
            # Stack and predict
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                if self.tta_wrapper:
                    # TTA for batch - process individually
                    batch_probs = []
                    for j in range(batch.size(0)):
                        probs = self.tta_wrapper.predict(batch[j:j+1])
                        batch_probs.append(probs)
                    probs = torch.cat(batch_probs, dim=0)
                else:
                    logits = self.model(batch)
                    logits = logits / self.temperature
                    probs = torch.softmax(logits, dim=1)
            
            # Process results
            for j, path in enumerate(valid_paths):
                pred_idx = probs[j].argmax().item()
                pred_conf = probs[j, pred_idx].item()
                
                result = {
                    'image': str(path),
                    'predicted_class': pred_idx,
                    'predicted_label': self.class_names[pred_idx],
                    'confidence': pred_conf
                }
                
                results.append(result)
        
        return results


def find_images(input_path: Path, extensions: List[str], recursive: bool = False) -> List[Path]:
    """Find all images in directory"""
    images = []
    
    if recursive:
        for ext in extensions:
            images.extend(input_path.rglob(f'*{ext}'))
            images.extend(input_path.rglob(f'*{ext.upper()}'))
    else:
        for ext in extensions:
            images.extend(input_path.glob(f'*{ext}'))
            images.extend(input_path.glob(f'*{ext.upper()}'))
    
    return sorted(list(set(images)))


def load_image_list_from_csv(csv_path: Path) -> List[Path]:
    """Load image paths from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Try to find image path column
    path_columns = ['image_path', 'path', 'filename', 'image', 'file']
    
    image_column = None
    for col in path_columns:
        if col in df.columns:
            image_column = col
            break
    
    if image_column is None:
        # Use first column
        image_column = df.columns[0]
        logger.warning(f"No standard image path column found, using '{image_column}'")
    
    # Get paths
    image_paths = df[image_column].tolist()
    
    # Convert to Path objects and check existence
    valid_paths = []
    for path_str in image_paths:
        path = Path(path_str)
        if path.exists():
            valid_paths.append(path)
        else:
            logger.warning(f"Image not found: {path}")
    
    logger.info(f"Loaded {len(valid_paths)} valid image paths from CSV")
    return valid_paths


def format_results(results: List[Dict[str, Any]], format_type: str,
                  include_probabilities: bool = False,
                  threshold: Optional[float] = None) -> Union[str, pd.DataFrame]:
    """Format results for output"""
    
    # Filter by threshold if specified
    if threshold is not None:
        results = [r for r in results if 'confidence' in r and r['confidence'] >= threshold]
    
    if format_type == 'csv':
        # Convert to DataFrame
        rows = []
        for r in results:
            row = {
                'image': r['image'],
                'predicted_label': r.get('predicted_label', 'ERROR'),
                'confidence': r.get('confidence', 0.0)
            }
            
            if 'error' in r:
                row['error'] = r['error']
            
            if include_probabilities and 'top_k_predictions' in r:
                for pred in r['top_k_predictions']:
                    row[f"prob_{pred['class_name']}"] = pred['probability']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    elif format_type == 'detailed':
        # Detailed text format
        lines = []
        lines.append("="*70)
        lines.append("RETFOUND PREDICTION RESULTS")
        lines.append("="*70)
        lines.append(f"Total images: {len(results)}")
        
        if threshold:
            lines.append(f"Confidence threshold: {threshold:.2%}")
        
        lines.append("")
        
        # Class distribution
        class_counts = {}
        for r in results:
            if 'predicted_label' in r:
                label = r['predicted_label']
                class_counts[label] = class_counts.get(label, 0) + 1
        
        lines.append("Predicted Class Distribution:")
        for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
        
        lines.append("\nDetailed Predictions:")
        lines.append("-"*70)
        
        for r in results:
            lines.append(f"\nImage: {r['image']}")
            if 'error' in r:
                lines.append(f"  ERROR: {r['error']}")
            else:
                lines.append(f"  Prediction: {r['predicted_label']} ({r['confidence']:.1%})")
                if 'top_k_predictions' in r and len(r['top_k_predictions']) > 1:
                    lines.append("  Top 5:")
                    for i, pred in enumerate(r['top_k_predictions'][:5]):
                        lines.append(f"    {i+1}. {pred['class_name']}: {pred['probability']:.1%}")
                if 'inference_time_ms' in r:
                    lines.append(f"  Inference time: {r['inference_time_ms']:.1f}ms")
        
        return '\n'.join(lines)
    
    else:  # json
        return results


def create_visualization(results: List[Dict[str, Any]], output_path: Path):
    """Create visualization of predictions"""
    import matplotlib.pyplot as plt
    from collections import Counter
    
    # Extract predictions
    predictions = [r['predicted_label'] for r in results if 'predicted_label' in r]
    
    if not predictions:
        logger.warning("No predictions to visualize")
        return
    
    # Count predictions
    counter = Counter(predictions)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    labels, counts = zip(*counter.most_common())
    x = range(len(labels))
    
    ax1.bar(x, counts)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Confidence distribution
    confidences = [r['confidence'] for r in results if 'confidence' in r]
    
    ax2.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'RETFound Predictions - {len(results)} Images', fontsize=16)
    plt.tight_layout()
    
    # Save
    vis_path = output_path.parent / f"{output_path.stem}_visualization.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {vis_path}")


def run_predict(args) -> int:
    """Main prediction function"""
    try:
        # Setup logging
        setup_logging()
        
        # Initialize predictor
        config = None
        if args.config:
            config = RETFoundConfig.load(Path(args.config))
        
        predictor = RETFoundPredictor(
            checkpoint_path=args.checkpoint,
            config=config,
            device=args.device,
            use_ema=args.use_ema
        )
        
        # Enable TTA if requested
        if args.use_tta:
            predictor.enable_tta(args.tta_augmentations)
        
        # Process input
        input_path = Path(args.input)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        
        # Collect images to process
        image_paths = []
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.csv':
                # CSV with image paths
                image_paths = load_image_list_from_csv(input_path)
            elif input_path.suffix.lower() in args.extensions:
                # Single image
                image_paths = [input_path]
            else:
                raise ValueError(f"Unknown file type: {input_path.suffix}")
        
        elif input_path.is_dir():
            # Directory of images
            image_paths = find_images(input_path, args.extensions, args.recursive)
            logger.info(f"Found {len(image_paths)} images in {input_path}")
        
        else:
            raise ValueError(f"Invalid input: {input_path}")
        
        if not image_paths:
            logger.error("No images found to process")
            return 1
        
        # Make predictions
        logger.info(f"Processing {len(image_paths)} images...")
        
        if len(image_paths) == 1:
            # Single image - detailed prediction
            results = [predictor.predict_single(image_paths[0], top_k=args.top_k)]
            results[0]['all_probabilities'] = None  # Add if needed
        else:
            # Batch prediction
            results = predictor.predict_batch(
                image_paths, 
                batch_size=args.batch_size,
                show_progress=True
            )
            
            # Add top-k predictions if requested
            if args.top_k > 1 or args.include_probabilities:
                logger.info("Adding detailed predictions...")
                for i, result in enumerate(results):
                    if 'error' not in result:
                        detailed = predictor.predict_single(
                            result['image'], 
                            top_k=args.top_k
                        )
                        result.update(detailed)
        
        # Determine output format
        if args.format == 'auto':
            if args.output and args.output.endswith('.csv'):
                output_format = 'csv'
            elif args.output and args.output.endswith('.json'):
                output_format = 'json'
            elif len(results) == 1:
                output_format = 'detailed'
            else:
                output_format = 'json'
        else:
            output_format = args.format
        
        # Format results
        formatted = format_results(
            results, 
            output_format,
            include_probabilities=args.include_probabilities,
            threshold=args.threshold
        )
        
        # Save or print results
        if args.output:
            output_path = Path(args.output)
            
            if output_format == 'csv':
                formatted.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
            elif output_format == 'detailed':
                with open(output_path, 'w') as f:
                    f.write(formatted)
                logger.info(f"Results saved to {output_path}")
            else:  # json
                with open(output_path, 'w') as f:
                    json.dump(formatted, f, indent=2)
                logger.info(f"Results saved to {output_path}")
            
            # Create visualization if requested
            if args.visualize and len(results) > 1:
                create_visualization(results, output_path)
        
        else:
            # Print to console
            if output_format == 'csv':
                print(formatted.to_string(index=False))
            elif output_format == 'detailed':
                print(formatted)
            else:
                print(json.dumps(formatted, indent=2))
        
        # Summary statistics
        if len(results) > 1:
            logger.info("\n" + "="*50)
            logger.info("SUMMARY")
            logger.info("="*50)
            
            success_results = [r for r in results if 'error' not in r]
            error_results = [r for r in results if 'error' in r]
            
            logger.info(f"Total images: {len(results)}")
            logger.info(f"Successful: {len(success_results)}")
            logger.info(f"Errors: {len(error_results)}")
            
            if success_results:
                confidences = [r['confidence'] for r in success_results]
                logger.info(f"Mean confidence: {np.mean(confidences):.3f}")
                logger.info(f"Min confidence: {np.min(confidences):.3f}")
                logger.info(f"Max confidence: {np.max(confidences):.3f}")
                
                if args.threshold:
                    above_threshold = sum(1 for c in confidences if c >= args.threshold)
                    logger.info(f"Above threshold ({args.threshold:.2%}): {above_threshold} "
                              f"({above_threshold/len(confidences)*100:.1f}%)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        logger.exception("Full traceback:")
        return 1
