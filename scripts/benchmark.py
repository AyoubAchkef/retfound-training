#!/usr/bin/env python3
"""
RETFound Performance Benchmark Script
=====================================

Benchmark various configurations and optimizations for RETFound training.
"""

import os
import sys
import time
import json
import argparse
import platform
import psutil
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.data.datasets import create_dummy_data
from retfound.training.optimizers import create_optimizer
from retfound.training.trainer import RETFoundTrainer
from retfound.utils.device import get_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    name: str
    config: Dict[str, Any]
    hardware: Dict[str, Any]
    timing: Dict[str, float]
    memory: Dict[str, float]
    throughput: Dict[str, float]
    errors: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for RETFound"""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            },
            'pytorch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        }
        
        # GPU information
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'memory_gb': props.total_memory / (1024**3),
                    'multi_processor_count': props.multi_processor_count,
                })
            info['gpu'] = gpu_info
            
        return info
    
    def benchmark_model_creation(self, config: RETFoundConfig) -> Tuple[float, float]:
        """Benchmark model creation time and memory"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_time = time.time()
        
        model = create_model(config)
        model.to(self.device)
        
        creation_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() - start_mem if torch.cuda.is_available() else 0
        
        del model
        torch.cuda.empty_cache()
        
        return creation_time, peak_mem / (1024**3)  # Convert to GB
    
    def benchmark_forward_pass(
        self, 
        config: RETFoundConfig, 
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_iterations: int = 50
    ) -> Dict[str, Any]:
        """Benchmark forward pass performance"""
        model = create_model(config)
        model.to(self.device)
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > 32 and config.input_size > 224:
                # Skip large batches for high resolution
                continue
                
            try:
                # Warmup
                dummy_input = torch.randn(
                    batch_size, 3, config.input_size, config.input_size,
                    device=self.device
                )
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(dummy_input)
                
                # Measure forward pass
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = model(dummy_input)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                total_time = time.time() - start_time
                
                # Calculate metrics
                time_per_batch = total_time / num_iterations
                images_per_second = batch_size / time_per_batch
                
                # Memory usage
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                
                results[f'batch_{batch_size}'] = {
                    'time_per_batch': time_per_batch,
                    'images_per_second': images_per_second,
                    'peak_memory_gb': peak_memory,
                }
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    results[f'batch_{batch_size}'] = {
                        'error': 'OOM',
                        'peak_memory_gb': torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
                    }
                else:
                    raise
            
            torch.cuda.empty_cache()
        
        del model
        return results
    
    def benchmark_training_step(
        self,
        config: RETFoundConfig,
        num_steps: int = 20
    ) -> Dict[str, Any]:
        """Benchmark complete training step"""
        # Create model and optimizer
        model = create_model(config)
        model.to(self.device)
        model.train()
        
        optimizer_config = {
            'name': 'adamw' if not config.use_sam else 'sam',
            'base_lr': config.base_lr,
            'weight_decay': config.weight_decay,
            'use_sam': config.use_sam,
            'sam_rho': config.sam_rho,
        }
        
        optimizer = create_optimizer(model, optimizer_config)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Dummy data
        dummy_input = torch.randn(
            config.batch_size, 3, config.input_size, config.input_size,
            device=self.device
        )
        dummy_target = torch.randint(
            0, config.num_classes, (config.batch_size,),
            device=self.device
        )
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        step_times = []
        for _ in range(num_steps):
            step_start = time.time()
            
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_times.append(time.time() - step_start)
        
        total_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() - start_mem if torch.cuda.is_available() else 0
        
        results = {
            'total_time': total_time,
            'avg_step_time': np.mean(step_times),
            'std_step_time': np.std(step_times),
            'min_step_time': np.min(step_times),
            'max_step_time': np.max(step_times),
            'peak_memory_gb': peak_mem / (1024**3),
            'throughput_img_per_sec': config.batch_size / np.mean(step_times),
        }
        
        del model, optimizer
        torch.cuda.empty_cache()
        
        return results
    
    def benchmark_optimization_features(self, base_config: RETFoundConfig) -> List[BenchmarkResult]:
        """Benchmark different optimization features"""
        optimization_configs = [
            ('baseline', {}),
            ('amp_fp16', {'use_amp': True, 'amp_dtype': torch.float16}),
            ('amp_bf16', {'use_amp': True, 'amp_dtype': torch.bfloat16}),
            ('gradient_checkpoint', {'use_gradient_checkpointing': True}),
            ('sam_optimizer', {'use_sam': True}),
            ('ema', {'use_ema': True}),
            ('compile', {'use_compile': True, 'compile_mode': 'default'}),
            ('all_optimizations', {
                'use_amp': True,
                'amp_dtype': torch.bfloat16,
                'use_gradient_checkpointing': True,
                'use_sam': True,
                'use_ema': True,
                'use_compile': True,
            }),
        ]
        
        results = []
        
        for name, opt_config in optimization_configs:
            logger.info(f"Benchmarking: {name}")
            
            # Update config
            config = RETFoundConfig(**{**asdict(base_config), **opt_config})
            
            try:
                # Skip compile on older PyTorch versions
                if config.use_compile and not hasattr(torch, 'compile'):
                    logger.warning(f"Skipping {name}: torch.compile not available")
                    continue
                
                # Model creation
                creation_time, creation_memory = self.benchmark_model_creation(config)
                
                # Forward pass
                forward_results = self.benchmark_forward_pass(
                    config, 
                    batch_sizes=[config.batch_size],
                    num_iterations=20
                )
                
                # Training step
                training_results = self.benchmark_training_step(config, num_steps=10)
                
                # Create result
                result = BenchmarkResult(
                    name=name,
                    config=opt_config,
                    hardware=self.get_system_info(),
                    timing={
                        'model_creation': creation_time,
                        'forward_pass': forward_results[f'batch_{config.batch_size}'].get('time_per_batch', -1),
                        'training_step': training_results['avg_step_time'],
                    },
                    memory={
                        'model_creation_gb': creation_memory,
                        'forward_pass_gb': forward_results[f'batch_{config.batch_size}'].get('peak_memory_gb', -1),
                        'training_step_gb': training_results['peak_memory_gb'],
                    },
                    throughput={
                        'forward_img_per_sec': forward_results[f'batch_{config.batch_size}'].get('images_per_second', -1),
                        'training_img_per_sec': training_results['throughput_img_per_sec'],
                    },
                    errors=[]
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error benchmarking {name}: {e}")
                result = BenchmarkResult(
                    name=name,
                    config=opt_config,
                    hardware=self.get_system_info(),
                    timing={},
                    memory={},
                    throughput={},
                    errors=[str(e)]
                )
                results.append(result)
            
            torch.cuda.empty_cache()
        
        return results
    
    def benchmark_batch_sizes(self, config: RETFoundConfig) -> List[BenchmarkResult]:
        """Benchmark different batch sizes"""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Update config
            config.batch_size = batch_size
            
            try:
                # Forward pass only (training might OOM)
                forward_results = self.benchmark_forward_pass(
                    config,
                    batch_sizes=[batch_size],
                    num_iterations=10
                )
                
                if f'batch_{batch_size}' in forward_results and 'error' not in forward_results[f'batch_{batch_size}']:
                    result = BenchmarkResult(
                        name=f'batch_size_{batch_size}',
                        config={'batch_size': batch_size},
                        hardware=self.get_system_info(),
                        timing={
                            'forward_pass': forward_results[f'batch_{batch_size}']['time_per_batch'],
                        },
                        memory={
                            'peak_memory_gb': forward_results[f'batch_{batch_size}']['peak_memory_gb'],
                        },
                        throughput={
                            'images_per_second': forward_results[f'batch_{batch_size}']['images_per_second'],
                        },
                        errors=[]
                    )
                else:
                    result = BenchmarkResult(
                        name=f'batch_size_{batch_size}',
                        config={'batch_size': batch_size},
                        hardware=self.get_system_info(),
                        timing={},
                        memory={
                            'peak_memory_gb': forward_results[f'batch_{batch_size}'].get('peak_memory_gb', -1),
                        },
                        throughput={},
                        errors=['OOM']
                    )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error with batch size {batch_size}: {e}")
                break
        
        return results
    
    def save_results(self, results: List[BenchmarkResult], name: str = "benchmark"):
        """Save benchmark results"""
        # Save as JSON
        json_path = self.output_dir / f"{name}_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            row = {
                'name': result.name,
                **{f'config_{k}': v for k, v in result.config.items()},
                **{f'timing_{k}': v for k, v in result.timing.items()},
                **{f'memory_{k}': v for k, v in result.memory.items()},
                **{f'throughput_{k}': v for k, v in result.throughput.items()},
                'errors': ','.join(result.errors) if result.errors else '',
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = self.output_dir / f"{name}_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
        
        return df
    
    def plot_results(self, results: List[BenchmarkResult], name: str = "benchmark"):
        """Create visualization plots for benchmark results"""
        df = self.save_results(results, name)
        
        # Setup plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'RETFound Benchmark Results - {name}', fontsize=16)
        
        # Plot 1: Training throughput
        if 'throughput_training_img_per_sec' in df.columns:
            ax = axes[0, 0]
            data = df[df['throughput_training_img_per_sec'] > 0]
            if not data.empty:
                sns.barplot(data=data, x='name', y='throughput_training_img_per_sec', ax=ax)
                ax.set_title('Training Throughput')
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Images/Second')
                ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Memory usage
        if 'memory_training_step_gb' in df.columns:
            ax = axes[0, 1]
            data = df[df['memory_training_step_gb'] > 0]
            if not data.empty:
                sns.barplot(data=data, x='name', y='memory_training_step_gb', ax=ax)
                ax.set_title('Training Memory Usage')
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Memory (GB)')
                ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Forward pass time
        if 'timing_forward_pass' in df.columns:
            ax = axes[1, 0]
            data = df[df['timing_forward_pass'] > 0]
            if not data.empty:
                sns.barplot(data=data, x='name', y='timing_forward_pass', ax=ax)
                ax.set_title('Forward Pass Time')
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Time (seconds)')
                ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Speedup comparison
        if 'throughput_training_img_per_sec' in df.columns:
            ax = axes[1, 1]
            data = df[df['throughput_training_img_per_sec'] > 0].copy()
            if not data.empty and 'baseline' in data['name'].values:
                baseline_throughput = data[data['name'] == 'baseline']['throughput_training_img_per_sec'].iloc[0]
                data['speedup'] = data['throughput_training_img_per_sec'] / baseline_throughput
                sns.barplot(data=data, x='name', y='speedup', ax=ax)
                ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
                ax.set_title('Speedup vs Baseline')
                ax.set_xlabel('Configuration')
                ax.set_ylabel('Speedup Factor')
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{name}_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {plot_path}")
    
    def generate_report(self, results: List[BenchmarkResult], name: str = "benchmark"):
        """Generate a comprehensive benchmark report"""
        report_path = self.output_dir / f"{name}_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RETFound Performance Benchmark Report\n")
            f.write("=" * 80 + "\n\n")
            
            # System information
            system_info = self.get_system_info()
            f.write("System Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Platform: {system_info['platform']['system']} {system_info['platform']['release']}\n")
            f.write(f"Python: {system_info['platform']['python_version']}\n")
            f.write(f"PyTorch: {system_info['pytorch']['version']}\n")
            
            if system_info['pytorch']['cuda_available']:
                f.write(f"CUDA: {system_info['pytorch']['cuda_version']}\n")
                f.write(f"GPU: {system_info['gpu'][0]['name']} ({system_info['gpu'][0]['memory_gb']:.1f} GB)\n")
            else:
                f.write("GPU: Not available\n")
            
            f.write(f"CPU: {system_info['hardware']['cpu_count']} cores\n")
            f.write(f"RAM: {system_info['hardware']['memory_total_gb']:.1f} GB\n")
            f.write("\n")
            
            # Results summary
            f.write("Benchmark Results Summary:\n")
            f.write("-" * 40 + "\n")
            
            # Find best configurations
            valid_results = [r for r in results if not r.errors and r.throughput.get('training_img_per_sec', 0) > 0]
            
            if valid_results:
                # Best throughput
                best_throughput = max(valid_results, key=lambda r: r.throughput.get('training_img_per_sec', 0))
                f.write(f"Best Training Throughput: {best_throughput.name}\n")
                f.write(f"  - {best_throughput.throughput['training_img_per_sec']:.1f} images/second\n")
                
                # Most memory efficient
                best_memory = min(valid_results, key=lambda r: r.memory.get('training_step_gb', float('inf')))
                f.write(f"\nMost Memory Efficient: {best_memory.name}\n")
                f.write(f"  - {best_memory.memory['training_step_gb']:.2f} GB\n")
                
                # Baseline comparison
                baseline = next((r for r in valid_results if r.name == 'baseline'), None)
                if baseline:
                    f.write(f"\nBaseline Performance:\n")
                    f.write(f"  - Throughput: {baseline.throughput.get('training_img_per_sec', 0):.1f} images/second\n")
                    f.write(f"  - Memory: {baseline.memory.get('training_step_gb', 0):.2f} GB\n")
                    
                    # Speedups
                    f.write(f"\nSpeedups vs Baseline:\n")
                    for result in valid_results:
                        if result.name != 'baseline':
                            speedup = result.throughput['training_img_per_sec'] / baseline.throughput['training_img_per_sec']
                            f.write(f"  - {result.name}: {speedup:.2f}x\n")
            
            # Errors
            error_results = [r for r in results if r.errors]
            if error_results:
                f.write(f"\nConfigurations with Errors:\n")
                for result in error_results:
                    f.write(f"  - {result.name}: {', '.join(result.errors)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Report saved to {report_path}")


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(
        description='Benchmark RETFound performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--suite',
        choices=['quick', 'optimizations', 'batch_sizes', 'full'],
        default='quick',
        help='Benchmark suite to run'
    )
    parser.add_argument(
        '--model-type',
        default='vit_large_patch16_224',
        help='Model architecture to benchmark'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Base batch size for benchmarks'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=22,
        help='Number of output classes'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='Input image size'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_results'),
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create base configuration
    base_config = RETFoundConfig(
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        input_size=args.input_size,
        epochs=1,  # Not used in benchmarks
        use_amp=False,
        use_sam=False,
        use_ema=False,
        use_compile=False,
        use_gradient_checkpointing=False,
    )
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(args.output)
    
    logger.info(f"Starting {args.suite} benchmark suite...")
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Input size: {args.input_size}x{args.input_size}")
    
    # Run benchmarks
    if args.suite == 'quick':
        # Quick benchmark with a few key configurations
        results = benchmark.benchmark_optimization_features(base_config)
        benchmark.plot_results(results, 'quick_benchmark')
        benchmark.generate_report(results, 'quick_benchmark')
        
    elif args.suite == 'optimizations':
        # Full optimization comparison
        results = benchmark.benchmark_optimization_features(base_config)
        benchmark.plot_results(results, 'optimization_benchmark')
        benchmark.generate_report(results, 'optimization_benchmark')
        
    elif args.suite == 'batch_sizes':
        # Batch size scaling
        results = benchmark.benchmark_batch_sizes(base_config)
        benchmark.plot_results(results, 'batch_size_benchmark')
        benchmark.generate_report(results, 'batch_size_benchmark')
        
    elif args.suite == 'full':
        # Complete benchmark suite
        opt_results = benchmark.benchmark_optimization_features(base_config)
        batch_results = benchmark.benchmark_batch_sizes(base_config)
        
        all_results = opt_results + batch_results
        benchmark.plot_results(all_results, 'full_benchmark')
        benchmark.generate_report(all_results, 'full_benchmark')
    
    logger.info(f"Benchmark complete! Results saved to {args.output}")


if __name__ == '__main__':
    main()