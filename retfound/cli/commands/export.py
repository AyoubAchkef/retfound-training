"""
Export Command
==============

CLI command for exporting trained RETFound models to various formats.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch
import torch.onnx
from datetime import datetime

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.export import ModelExporter
from retfound.export.onnx import ONNXExporter
from retfound.export.torchscript import TorchScriptExporter
from retfound.export.tensorrt import TensorRTExporter
from retfound.utils.logging import setup_logging
from retfound.utils.device import get_device

logger = logging.getLogger(__name__)


def add_export_args(parser):
    """Add export-specific arguments to parser"""
    parser.add_argument('checkpoint', type=str,
                       help='Path to model checkpoint to export')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save exported models')
    parser.add_argument('--formats', nargs='+', 
                       default=['pytorch', 'torchscript', 'onnx'],
                       choices=['pytorch', 'torchscript', 'onnx', 'tensorrt', 'all'],
                       help='Export formats')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if different from checkpoint)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for export (affects dynamic shapes)')
    parser.add_argument('--opset-version', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify ONNX model')
    parser.add_argument('--fp16', action='store_true',
                       help='Export with FP16 precision')
    parser.add_argument('--int8', action='store_true',
                       help='Export with INT8 quantization')
    parser.add_argument('--dynamic-batch', action='store_true',
                       help='Export with dynamic batch size')
    parser.add_argument('--max-batch-size', type=int, default=32,
                       help='Maximum batch size for dynamic export')
    parser.add_argument('--optimize-for-mobile', action='store_true',
                       help='Optimize exported model for mobile deployment')
    parser.add_argument('--include-preprocessing', action='store_true',
                       help='Include preprocessing in exported model')
    parser.add_argument('--use-ema', action='store_true',
                       help='Use EMA model if available')
    parser.add_argument('--create-model-card', action='store_true',
                       help='Create model card documentation')
    parser.add_argument('--create-inference-script', action='store_true',
                       help='Create standalone inference script')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported models')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for export')


def load_checkpoint_for_export(checkpoint_path: Path, config: RETFoundConfig,
                              use_ema: bool = False) -> Dict[str, Any]:
    """Load checkpoint and prepare for export"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = create_model(config)
    
    # Load weights
    if use_ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        logger.info("Loading EMA model weights")
        model.load_state_dict(checkpoint['ema_state_dict'])
        model_type = "ema"
    else:
        logger.info("Loading regular model weights")
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = "regular"
    
    # Get metadata
    metadata = {
        'model_type': model_type,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_acc': checkpoint.get('best_val_acc', 0),
        'best_val_auc': checkpoint.get('best_val_auc', 0),
        'num_classes': config.num_classes,
        'input_size': config.input_size,
        'architecture': config.model_type,
        'pretrained_weights': checkpoint.get('config', {}).get('weights_key', 'unknown'),
        'training_time_hours': sum(checkpoint.get('history', {}).get('epoch_time', [])) / 3600 if 'history' in checkpoint else 0,
        'temperature': checkpoint.get('temperature', None),
        'class_names': checkpoint.get('class_names', [f'Class_{i}' for i in range(config.num_classes)]),
        'normalization': {
            'mean': config.pixel_mean,
            'std': config.pixel_std
        }
    }
    
    return model, metadata


def export_pytorch(model: torch.nn.Module, output_dir: Path, metadata: Dict[str, Any]):
    """Export in PyTorch format with full metadata"""
    logger.info("Exporting PyTorch model...")
    
    export_path = output_dir / 'retfound_model.pth'
    
    # Save complete model package
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'export_info': {
            'export_date': datetime.now().isoformat(),
            'export_version': '2.0',
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
    }, export_path)
    
    logger.info(f"PyTorch model saved to {export_path}")
    
    # Also save just the state dict for easier loading
    state_dict_path = output_dir / 'retfound_weights.pth'
    torch.save(model.state_dict(), state_dict_path)
    logger.info(f"Model weights saved to {state_dict_path}")
    
    return export_path


def create_model_card(output_dir: Path, metadata: Dict[str, Any], 
                     export_formats: list, validation_results: Optional[Dict] = None):
    """Create comprehensive model card"""
    card = []
    card.append("# RETFound Model Card\n")
    card.append(f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    card.append("## Model Details\n")
    card.append(f"- **Architecture**: {metadata['architecture']}")
    card.append(f"- **Number of Classes**: {metadata['num_classes']}")
    card.append(f"- **Input Size**: {metadata['input_size']}x{metadata['input_size']}")
    card.append(f"- **Pre-trained Weights**: {metadata['pretrained_weights']}")
    card.append(f"- **Model Type**: {metadata['model_type']}")
    card.append(f"- **Training Epochs**: {metadata['epoch']}")
    card.append(f"- **Training Time**: {metadata['training_time_hours']:.1f} hours\n")
    
    card.append("## Performance\n")
    card.append(f"- **Best Validation Accuracy**: {metadata['best_val_acc']:.2f}%")
    card.append(f"- **Best Validation AUC**: {metadata['best_val_auc']:.4f}\n")
    
    if metadata.get('temperature'):
        card.append(f"- **Temperature Scaling**: {metadata['temperature']:.3f}\n")
    
    card.append("## Classes\n")
    for i, class_name in enumerate(metadata['class_names']):
        card.append(f"{i}. {class_name}")
    card.append("")
    
    card.append("## Input Preprocessing\n")
    card.append("```python")
    card.append("# Normalization")
    card.append(f"mean = {metadata['normalization']['mean']}")
    card.append(f"std = {metadata['normalization']['std']}")
    card.append("```\n")
    
    card.append("## Export Information\n")
    card.append(f"- **Export Formats**: {', '.join(export_formats)}")
    card.append(f"- **PyTorch Version**: {torch.__version__}")
    if torch.cuda.is_available():
        card.append(f"- **CUDA Version**: {torch.version.cuda}")
    card.append("")
    
    if validation_results:
        card.append("## Validation Results\n")
        for format_name, results in validation_results.items():
            card.append(f"\n### {format_name}")
            card.append(f"- **Validation**: {'✅ Passed' if results['valid'] else '❌ Failed'}")
            if results.get('max_diff') is not None:
                card.append(f"- **Max Difference**: {results['max_diff']:.6f}")
            if results.get('inference_time') is not None:
                card.append(f"- **Inference Time**: {results['inference_time']:.3f}ms")
    
    card.append("\n## Usage\n")
    card.append("### PyTorch\n")
    card.append("```python")
    card.append("import torch")
    card.append("from retfound.models import RETFoundModel")
    card.append("")
    card.append("# Load model")
    card.append("checkpoint = torch.load('retfound_model.pth')")
    card.append("model = RETFoundModel(config)")
    card.append("model.load_state_dict(checkpoint['model_state_dict'])")
    card.append("model.eval()")
    card.append("```\n")
    
    if 'torchscript' in export_formats:
        card.append("### TorchScript\n")
        card.append("```python")
        card.append("import torch")
        card.append("")
        card.append("# Load traced model")
        card.append("model = torch.jit.load('retfound_traced.pt')")
        card.append("model.eval()")
        card.append("```\n")
    
    if 'onnx' in export_formats:
        card.append("### ONNX\n")
        card.append("```python")
        card.append("import onnxruntime as ort")
        card.append("")
        card.append("# Create session")
        card.append("session = ort.InferenceSession('retfound.onnx')")
        card.append("")
        card.append("# Run inference")
        card.append("inputs = {session.get_inputs()[0].name: image_tensor}")
        card.append("outputs = session.run(None, inputs)")
        card.append("```\n")
    
    card.append("## Limitations\n")
    card.append("- This model is for research purposes only")
    card.append("- Clinical validation required before medical use")
    card.append("- Performance may vary on different populations")
    card.append("- Regular monitoring and updates recommended\n")
    
    card.append("## Citation\n")
    card.append("```bibtex")
    card.append("@article{retfound2023,")
    card.append("  title={A foundation model for generalizable disease detection from retinal images},")
    card.append("  author={Zhou, Yukun and others},")
    card.append("  journal={Nature},")
    card.append("  year={2023}")
    card.append("}")
    card.append("```")
    
    # Save model card
    card_path = output_dir / 'MODEL_CARD.md'
    with open(card_path, 'w') as f:
        f.write('\n'.join(card))
    
    logger.info(f"Model card saved to {card_path}")


def create_inference_script(output_dir: Path, metadata: Dict[str, Any]):
    """Create standalone inference script"""
    script = '''#!/usr/bin/env python3
"""
RETFound Inference Script
Auto-generated for model deployment
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import json
import time


class RETFoundInference:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Model metadata
        self.metadata = {metadata}
        self.class_names = self.metadata['class_names']
        self.input_size = self.metadata['input_size']
        self.mean = np.array(self.metadata['normalization']['mean'])
        self.std = np.array(self.metadata['normalization']['std'])
        self.temperature = self.metadata.get('temperature', 1.0)
        
        # Load model based on format
        model_path = Path(model_path)
        if model_path.suffix == '.pt':
            # TorchScript
            self.model = torch.jit.load(str(model_path), map_location=self.device)
        elif model_path.suffix == '.pth':
            # PyTorch checkpoint - requires model definition
            raise NotImplementedError("PyTorch checkpoint loading requires model definition")
        else:
            raise ValueError(f"Unknown model format: {model_path.suffix}")
        
        self.model.eval()
    
    def preprocess(self, image_path):
        """Preprocess image for inference"""
        # Load and resize
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.input_size, self.input_size), Image.LANCZOS)
        
        # Convert to numpy and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0).to(self.device)
        
        return image
    
    def predict(self, image_path, top_k=5):
        """Predict class for image"""
        # Preprocess
        image = self.preprocess(image_path)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(image)
            logits = logits / self.temperature
            probs = F.softmax(logits, dim=1)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(self.class_names)))
        
        results = {
            'top_predictions': [],
            'all_probabilities': probs[0].cpu().numpy().tolist(),
            'inference_time_ms': inference_time
        }
        
        for i in range(len(top_indices)):
            results['top_predictions'].append({
                'class_id': top_indices[i].item(),
                'class_name': self.class_names[top_indices[i].item()],
                'probability': top_probs[i].item()
            })
        
        return results
    
    def predict_batch(self, image_paths, batch_size=32):
        """Predict for multiple images"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = torch.cat([self.preprocess(p) for p in batch_paths])
            
            with torch.no_grad():
                logits = self.model(batch_images)
                logits = logits / self.temperature
                probs = F.softmax(logits, dim=1)
            
            for j, path in enumerate(batch_paths):
                pred_idx = probs[j].argmax().item()
                results.append({
                    'image': str(path),
                    'prediction': self.class_names[pred_idx],
                    'confidence': probs[j, pred_idx].item(),
                    'probabilities': probs[j].cpu().numpy().tolist()
                })
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RETFound Inference')
    parser.add_argument('image', help='Image path or directory')
    parser.add_argument('--model', default='retfound_traced.pt', help='Model path')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model from {args.model}...")
    inferencer = RETFoundInference(args.model, device=args.device)
    print(f"Model loaded on {inferencer.device}")
    
    # Process input
    input_path = Path(args.image)
    
    if input_path.is_file():
        # Single image
        print(f"Processing {input_path.name}...")
        results = inferencer.predict(str(input_path), top_k=args.top_k)
        
        print(f"\\nTop {args.top_k} predictions:")
        for pred in results['top_predictions']:
            print(f"  {pred['class_name']}: {pred['probability']:.1%}")
        print(f"\\nInference time: {results['inference_time_ms']:.1f}ms")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    
    elif input_path.is_dir():
        # Directory
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
        
        print(f"Processing {len(image_paths)} images...")
        results = inferencer.predict_batch(image_paths, batch_size=args.batch_size)
        
        # Summary
        print(f"\\nProcessed {len(results)} images")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for r in results[:5]:  # Show first 5
                print(f"  {Path(r['image']).name}: {r['prediction']} ({r['confidence']:.1%})")
            if len(results) > 5:
                print(f"  ... and {len(results)-5} more")
    
    else:
        print(f"Error: {input_path} not found")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
'''
    
    # Replace metadata placeholder
    import json
    script = script.replace('{metadata}', json.dumps(metadata, indent=8))
    
    # Save script
    script_path = output_dir / 'inference.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Make executable
    script_path.chmod(0o755)
    
    logger.info(f"Inference script saved to {script_path}")


def validate_export(model_path: Path, original_model: torch.nn.Module, 
                   config: RETFoundConfig, device: torch.device) -> Dict[str, Any]:
    """Validate exported model against original"""
    logger.info(f"Validating {model_path.name}...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, config.input_size, config.input_size).to(device)
    
    # Get original output
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(dummy_input)
    
    try:
        if model_path.suffix == '.pt':
            # TorchScript
            traced_model = torch.jit.load(str(model_path))
            traced_model.eval()
            with torch.no_grad():
                export_output = traced_model(dummy_input)
        
        elif model_path.suffix == '.onnx':
            # ONNX
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(model_path))
            inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            export_output = session.run(None, inputs)[0]
            export_output = torch.from_numpy(export_output)
        
        else:
            return {'valid': False, 'error': f'Unknown format: {model_path.suffix}'}
        
        # Compare outputs
        max_diff = (original_output.cpu() - export_output.cpu()).abs().max().item()
        
        # Measure inference time
        import time
        times = []
        for _ in range(10):
            start = time.time()
            if model_path.suffix == '.pt':
                with torch.no_grad():
                    _ = traced_model(dummy_input)
            else:
                _ = session.run(None, inputs)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times[2:])  # Skip warmup
        
        return {
            'valid': max_diff < 1e-3,
            'max_diff': max_diff,
            'inference_time': avg_time,
            'format': model_path.suffix
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def run_export(args) -> int:
    """Main export function"""
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
        elif 'config' in checkpoint:
            config = RETFoundConfig(**checkpoint['config'])
        else:
            raise ValueError("No configuration found. Please provide --config")
        
        # Override batch size if specified
        if args.batch_size:
            config.batch_size = args.batch_size
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Load model and metadata
        model, metadata = load_checkpoint_for_export(
            checkpoint_path, config, use_ema=args.use_ema
        )
        model = model.to(device)
        model.eval()
        
        # Process export formats
        if 'all' in args.formats:
            formats = ['pytorch', 'torchscript', 'onnx', 'tensorrt']
        else:
            formats = args.formats
        
        logger.info(f"Exporting to formats: {formats}")
        
        exported_files = {}
        validation_results = {}
        
        # Create exporters
        exporters = {
            'torchscript': TorchScriptExporter(optimize=not args.optimize_for_mobile),
            'onnx': ONNXExporter(
                opset_version=args.opset_version,
                simplify=args.simplify,
                dynamic_batch=args.dynamic_batch,
                max_batch_size=args.max_batch_size
            )
        }
        
        if 'tensorrt' in formats:
            exporters['tensorrt'] = TensorRTExporter(
                fp16=args.fp16,
                int8=args.int8,
                max_batch_size=args.max_batch_size
            )
        
        # Export PyTorch format
        if 'pytorch' in formats:
            exported_files['pytorch'] = export_pytorch(model, output_dir, metadata)
        
        # Export other formats
        exporter = ModelExporter(model, config, exporters)
        
        for format_name in formats:
            if format_name == 'pytorch':
                continue
            
            logger.info(f"\nExporting to {format_name}...")
            try:
                export_path = exporter.export(
                    format_name,
                    output_dir,
                    input_shape=(config.batch_size, 3, config.input_size, config.input_size),
                    metadata=metadata
                )
                exported_files[format_name] = export_path
                logger.info(f"Exported to {export_path}")
                
                # Validate if requested
                if args.validate:
                    val_result = validate_export(export_path, model, config, device)
                    validation_results[format_name] = val_result
                    
                    if val_result['valid']:
                        logger.info(f"✅ Validation passed - Max diff: {val_result['max_diff']:.6f}, "
                                  f"Inference: {val_result['inference_time']:.1f}ms")
                    else:
                        logger.error(f"❌ Validation failed: {val_result.get('error', 'Unknown error')}")
                        
            except Exception as e:
                logger.error(f"Failed to export {format_name}: {e}")
                if args.validate:
                    validation_results[format_name] = {'valid': False, 'error': str(e)}
        
        # Create model card
        if args.create_model_card:
            create_model_card(output_dir, metadata, list(exported_files.keys()), 
                            validation_results if args.validate else None)
        
        # Create inference script
        if args.create_inference_script:
            create_inference_script(output_dir, metadata)
        
        # Save export summary
        summary = {
            'export_date': datetime.now().isoformat(),
            'checkpoint': str(checkpoint_path),
            'formats': list(exported_files.keys()),
            'files': {k: str(v) for k, v in exported_files.items()},
            'metadata': metadata,
            'validation': validation_results if args.validate else None
        }
        
        summary_path = output_dir / 'export_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("EXPORT COMPLETED")
        logger.info("="*70)
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Exported formats: {', '.join(exported_files.keys())}")
        
        if args.validate:
            logger.info("\nValidation Results:")
            for fmt, result in validation_results.items():
                status = "✅ PASSED" if result['valid'] else "❌ FAILED"
                logger.info(f"  {fmt}: {status}")
        
        logger.info(f"\nAll files saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Export failed with error: {e}")
        logger.exception("Full traceback:")
        return 1
