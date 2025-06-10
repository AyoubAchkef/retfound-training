#!/usr/bin/env python3
"""
Benchmark script for RETFound models on CAASI dataset v6.1.
Supports benchmarking speed, memory usage, and accuracy across different configurations.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil
import GPUtil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retfound.core.config import RETFoundConfig
from retfound.data import create_datamodule
from retfound.models import create_model
from retfound.metrics.medical import OphthalmologyMetrics
from retfound.utils.device import get_device
from retfound.utils.logging import get_logger

logger = get_logger(__name__)


class RETFoundBenchmark:
    """Comprehensive benchmarking for RETFound models."""
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "./benchmark_results",
        device: Optional[torch.device] = None,
        dataset_version: str = "v6.1"
    ):
        """
        Initialize benchmark suite.
        
        Args:
            dataset_path: Path to dataset
            output_dir: Directory to save results
            device: Device for benchmarking
            dataset_version: Dataset version to use
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()
        self.dataset_version = dataset_version
        
        # Results storage
        self.results = {
            'system_info': self._get_system_info(),
            'dataset_info': {
                'version': dataset_version,
                'path': str(dataset_path)
            },
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            
            # Get GPU memory
            gpus = GPUtil.getGPUs()
            info['gpu_memory_gb'] = [gpu.memoryTotal / 1024 for gpu in gpus]
        
        return info
    
    def benchmark_model(
        self,
        model_type: str = "vit_large_patch16_224",
        batch_sizes: List[int] = [1, 8, 16, 32],
        input_sizes: List[int] = [224],
        num_classes: int = 28,
        pretrained_weights: Optional[str] = None,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        test_accuracy: bool = True
    ) -> Dict:
        """
        Benchmark a specific model configuration.
        
        Args:
            model_type: Model architecture to benchmark
            batch_sizes: List of batch sizes to test
            input_sizes: List of input sizes to test
            num_classes: Number of output classes
            pretrained_weights: Optional pretrained weights path
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            test_accuracy: Whether to test accuracy on validation set
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking {model_type}...")
        
        results = {
            'model_type': model_type,
            'num_classes': num_classes,
            'configurations': []
        }
        
        for input_size in input_sizes:
            for batch_size in batch_sizes:
                logger.info(f"Testing input_size={input_size}, batch_size={batch_size}")
                
                try:
                    config_results = self._benchmark_configuration(
                        model_type=model_type,
                        batch_size=batch_size,
                        input_size=input_size,
                        num_classes=num_classes,
                        pretrained_weights=pretrained_weights,
                        warmup_iterations=warmup_iterations,
                        benchmark_iterations=benchmark_iterations,
                        test_accuracy=test_accuracy
                    )
                    results['configurations'].append(config_results)
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark configuration: {str(e)}")
                    results['configurations'].append({
                        'input_size': input_size,
                        'batch_size': batch_size,
                        'error': str(e)
                    })
                
                # Clear cache
                torch.cuda.empty_cache()
        
        return results
    
    def _benchmark_configuration(
        self,
        model_type: str,
        batch_size: int,
        input_size: int,
        num_classes: int,
        pretrained_weights: Optional[str],
        warmup_iterations: int,
        benchmark_iterations: int,
        test_accuracy: bool
    ) -> Dict:
        """Benchmark a specific configuration."""
        # Create model
        config = RETFoundConfig(
            model_type=model_type,
            num_classes=num_classes,
            input_size=input_size,
            batch_size=batch_size
        )
        
        model = create_model(config)
        
        # Load pretrained weights if provided
        if pretrained_weights:
            checkpoint = torch.load(pretrained_weights, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(self.device)
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Benchmark inference speed
        logger.info("Benchmarking inference speed...")
        inference_times = []
        memory_usage = []
        
        for _ in tqdm(range(benchmark_iterations), desc="Inference"):
            # Record memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            
            # Time inference
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Record memory after
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append((mem_after - mem_before) / (1024**2))  # MB
            
            inference_times.append(end_time - start_time)
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        throughput = batch_size / inference_times.mean()
        
        results = {
            'input_size': input_size,
            'batch_size': batch_size,
            'inference_time_ms': {
                'mean': inference_times.mean() * 1000,
                'std': inference_times.std() * 1000,
                'min': inference_times.min() * 1000,
                'max': inference_times.max() * 1000,
                'p50': np.percentile(inference_times, 50) * 1000,
                'p95': np.percentile(inference_times, 95) * 1000,
                'p99': np.percentile(inference_times, 99) * 1000
            },
            'throughput_fps': throughput,
            'latency_per_image_ms': (inference_times.mean() * 1000) / batch_size
        }
        
        if memory_usage:
            results['memory_mb'] = {
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage)
            }
        
        # Test accuracy if requested
        if test_accuracy:
            logger.info("Testing accuracy on validation set...")
            accuracy_results = self._test_accuracy(model, config)
            results['accuracy'] = accuracy_results
        
        # Get model info
        results['model_info'] = self._get_model_info(model)
        
        return results
    
    def _test_accuracy(self, model: torch.nn.Module, config: RETFoundConfig) -> Dict:
        """Test model accuracy on validation set."""
        # Create datamodule
        data_config = config.to_dict()
        data_config['data'] = {
            'dataset_path': str(self.dataset_path),
            'dataset_version': self.dataset_version,
            'batch_size': config.batch_size,
            'num_workers': 4
        }
        
        datamodule = create_datamodule(data_config)
        datamodule.setup('fit')
        val_loader = datamodule.val_dataloader()
        
        # Initialize metrics
        metrics = OphthalmologyMetrics(
            num_classes=config.num_classes,
            dataset_version=self.dataset_version
        )
        
        # Run validation
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                metrics.update(preds, labels, probs)
        
        # Compute metrics
        results = metrics.compute()
        
        # Return key metrics
        return {
            'accuracy': results.get('accuracy', 0),
            'balanced_accuracy': results.get('balanced_accuracy', 0),
            'f1_macro': results.get('f1_macro', 0),
            'auc_macro': results.get('auc_macro', 0)
        }
    
    def _get_model_info(self, model: torch.nn.Module) -> Dict:
        """Get model information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / (1024**2)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def benchmark_all_models(
        self,
        model_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Benchmark multiple model types.
        
        Args:
            model_types: List of model types to benchmark
            **kwargs: Additional arguments for benchmark_model
            
        Returns:
            All benchmark results
        """
        if model_types is None:
            # Default models to benchmark
            model_types = [
                "vit_tiny_patch16_224",
                "vit_small_patch16_224", 
                "vit_base_patch16_224",
                "vit_large_patch16_224",
                "resnet18",
                "resnet50",
                "efficientnet_b0",
                "efficientnet_b3"
            ]
        
        for model_type in model_types:
            logger.info(f"\nBenchmarking {model_type}")
            logger.info("=" * 60)
            
            results = self.benchmark_model(model_type=model_type, **kwargs)
            self.results['benchmarks'][model_type] = results
            
            # Save intermediate results
            self._save_results()
        
        return self.results
    
    def compare_modalities(
        self,
        model_type: str = "vit_large_patch16_224",
        batch_size: int = 16
    ) -> Dict:
        """
        Compare performance on fundus vs OCT modalities.
        
        Args:
            model_type: Model type to test
            batch_size: Batch size for testing
            
        Returns:
            Comparison results
        """
        logger.info("Comparing modalities...")
        
        results = {
            'model_type': model_type,
            'batch_size': batch_size,
            'modalities': {}
        }
        
        for modality in ['fundus', 'oct', 'both']:
            logger.info(f"Testing {modality} modality...")
            
            # Create model with appropriate number of classes
            if modality == 'fundus':
                num_classes = 18
            elif modality == 'oct':
                num_classes = 10
            else:
                num_classes = 28
            
            # Benchmark
            modality_results = self._benchmark_configuration(
                model_type=model_type,
                batch_size=batch_size,
                input_size=224,
                num_classes=num_classes,
                pretrained_weights=None,
                warmup_iterations=5,
                benchmark_iterations=50,
                test_accuracy=False
            )
            
            results['modalities'][modality] = modality_results
        
        self.results['benchmarks']['modality_comparison'] = results
        self._save_results()
        
        return results
    
    def _save_results(self):
        """Save benchmark results to file."""
        output_file = self.output_dir / "benchmark_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_report(self):
        """Generate benchmark report."""
        report_path = self.output_dir / "benchmark_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("RETFound Benchmark Report\n")
            f.write("=" * 80 + "\n\n")
            
            # System info
            f.write("System Information:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.results['system_info'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Benchmark results
            f.write("Benchmark Results:\n")
            f.write("-" * 40 + "\n")
            
            for model_name, model_results in self.results['benchmarks'].items():
                f.write(f"\n{model_name}:\n")
                
                if 'configurations' in model_results:
                    for config in model_results['configurations']:
                        if 'error' in config:
                            f.write(f"  Error: {config['error']}\n")
                            continue
                        
                        f.write(f"  Batch size {config['batch_size']}:\n")
                        f.write(f"    Throughput: {config['throughput_fps']:.2f} FPS\n")
                        f.write(f"    Latency: {config['latency_per_image_ms']:.2f} ms/image\n")
                        
                        if 'memory_mb' in config:
                            f.write(f"    Memory: {config['memory_mb']['mean']:.2f} MB\n")
                        
                        if 'accuracy' in config:
                            f.write(f"    Accuracy: {config['accuracy']['accuracy']:.4f}\n")
        
        logger.info(f"Report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark RETFound models on CAASI dataset v6.1"
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Model types to benchmark'
    )
    parser.add_argument(
        '--batch-sizes',
        nargs='+',
        type=int,
        default=[1, 8, 16, 32],
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--test-accuracy',
        action='store_true',
        help='Test accuracy on validation set'
    )
    parser.add_argument(
        '--compare-modalities',
        action='store_true',
        help='Compare performance on different modalities'
    )
    parser.add_argument(
        '--dataset-version',
        type=str,
        default='v6.1',
        choices=['v4.0', 'v6.1'],
        help='Dataset version'
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = RETFoundBenchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version
    )
    
    # Run benchmarks
    if args.compare_modalities:
        benchmark.compare_modalities()
    else:
        benchmark.benchmark_all_models(
            model_types=args.models,
            batch_sizes=args.batch_sizes,
            test_accuracy=args.test_accuracy
        )
    
    # Generate report
    benchmark.generate_report()


if __name__ == '__main__':
    main()