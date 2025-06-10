"""
TorchScript Export
==================

Export models to TorchScript format for optimized inference.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.jit import ScriptModule

logger = logging.getLogger(__name__)


class TorchScriptExporter:
    """
    Export PyTorch models to TorchScript format
    """
    
    def __init__(self, config: Any):
        """
        Initialize TorchScript exporter
        
        Args:
            config: Export configuration
        """
        self.config = config
    
    def export(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export model to TorchScript format
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            output_path: Output file path
            
        Returns:
            Path to exported TorchScript model
        """
        if output_path is None:
            suffix = "_traced" if self.config.torchscript_trace else "_scripted"
            output_path = self.config.output_dir / f"{self.config.model_name}{suffix}.pt"
        
        logger.info(f"Exporting to TorchScript: {output_path}")
        
        # Prepare model
        model.eval()
        
        # Export using tracing or scripting
        if self.config.torchscript_trace:
            scripted_model = self._trace_model(model, example_input)
        else:
            scripted_model = self._script_model(model)
        
        # Optimize if requested
        if self.config.optimize:
            scripted_model = self._optimize_model(scripted_model)
        
        # Save model
        scripted_model.save(str(output_path))
        
        # Verify saved model
        try:
            loaded_model = torch.jit.load(str(output_path))
            with torch.no_grad():
                _ = loaded_model(example_input)
            logger.info("TorchScript model verification passed")
        except Exception as e:
            logger.error(f"TorchScript verification failed: {e}")
            raise
        
        logger.info(f"TorchScript export completed: {output_path}")
        
        return output_path
    
    def _trace_model(self, model: nn.Module, example_input: torch.Tensor) -> ScriptModule:
        """Trace model execution"""
        logger.info("Tracing model...")
        
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                example_input,
                check_trace=True,
                strict=False
            )
        
        return traced_model
    
    def _script_model(self, model: nn.Module) -> ScriptModule:
        """Script model (requires torchscript annotations)"""
        logger.info("Scripting model...")
        
        try:
            scripted_model = torch.jit.script(model)
            return scripted_model
        except Exception as e:
            logger.warning(f"Scripting failed: {e}. Falling back to tracing.")
            # Create dummy input for tracing
            device = next(model.parameters()).device
            dummy_input = torch.randn(
                self.config.batch_size,
                *self.config.input_shape,
                device=device
            )
            return self._trace_model(model, dummy_input)
    
    def _optimize_model(self, model: ScriptModule) -> ScriptModule:
        """Apply TorchScript optimizations"""
        logger.info("Optimizing TorchScript model...")
        
        # Freeze model (inline parameters)
        if hasattr(model, '_c'):
            torch.jit.freeze(model)
        
        # Apply optimization passes
        torch.jit.optimize_for_inference(model)
        
        return model
    
    def verify(
        self,
        model_path: Path,
        example_input: torch.Tensor,
        expected_output: torch.Tensor,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Verify TorchScript model outputs match PyTorch
        
        Args:
            model_path: Path to TorchScript model
            example_input: Example input tensor
            expected_output: Expected output from PyTorch model
            tolerance: Numerical tolerance
            
        Returns:
            True if outputs match within tolerance
        """
        try:
            # Load TorchScript model
            scripted_model = torch.jit.load(str(model_path))
            scripted_model.eval()
            
            # Run inference
            with torch.no_grad():
                script_output = scripted_model(example_input)
            
            # Compare outputs
            is_close = torch.allclose(
                expected_output,
                script_output,
                rtol=tolerance,
                atol=tolerance
            )
            
            if is_close:
                max_diff = torch.max(torch.abs(expected_output - script_output))
                logger.info(f"TorchScript verification passed. Max difference: {max_diff:.6f}")
            else:
                logger.warning("TorchScript verification failed. Outputs do not match.")
            
            return is_close
            
        except Exception as e:
            logger.error(f"TorchScript verification error: {e}")
            return False


def trace_model(
    model: nn.Module,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: Union[str, Path],
    optimize: bool = True,
    check_trace: bool = True
) -> ScriptModule:
    """
    Trace a PyTorch model to TorchScript
    
    Args:
        model: Model to trace
        example_inputs: Example inputs for tracing
        output_path: Path to save traced model
        optimize: Whether to optimize the traced model
        check_trace: Whether to verify trace
        
    Returns:
        Traced ScriptModule
    """
    model.eval()
    
    # Ensure inputs are tuple
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)
    
    # Trace model
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            example_inputs,
            check_trace=check_trace,
            strict=False
        )
    
    # Optimize if requested
    if optimize:
        traced = optimize_torchscript(traced)
    
    # Save
    output_path = Path(output_path)
    traced.save(str(output_path))
    
    logger.info(f"Traced model saved to {output_path}")
    
    return traced


def script_model(
    model: nn.Module,
    output_path: Union[str, Path],
    optimize: bool = True
) -> ScriptModule:
    """
    Script a PyTorch model to TorchScript
    
    Args:
        model: Model to script (must have torchscript annotations)
        output_path: Path to save scripted model
        optimize: Whether to optimize the scripted model
        
    Returns:
        Scripted ScriptModule
    """
    model.eval()
    
    # Script model
    scripted = torch.jit.script(model)
    
    # Optimize if requested
    if optimize:
        scripted = optimize_torchscript(scripted)
    
    # Save
    output_path = Path(output_path)
    scripted.save(str(output_path))
    
    logger.info(f"Scripted model saved to {output_path}")
    
    return scripted


def optimize_torchscript(model: ScriptModule) -> ScriptModule:
    """
    Apply optimizations to TorchScript model
    
    Args:
        model: TorchScript model
        
    Returns:
        Optimized model
    """
    # Freeze model (inline parameters and attributes)
    if hasattr(torch.jit, 'freeze'):
        model = torch.jit.freeze(model)
    
    # Optimize for inference
    if hasattr(torch.jit, 'optimize_for_inference'):
        model = torch.jit.optimize_for_inference(model)
    
    return model


def profile_torchscript(
    model_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Profile TorchScript model performance
    
    Args:
        model_path: Path to TorchScript model
        input_shape: Input tensor shape
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        device: Device to run on
        
    Returns:
        Profiling statistics
    """
    import time
    import numpy as np
    
    # Load model
    model = torch.jit.load(str(model_path))
    model.eval()
    model.to(device)
    
    # Create input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    logger.info(f"Warming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    logger.info(f"Profiling with {num_runs} runs...")
    times = []
    
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    stats = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'throughput_fps': 1000 / np.mean(times)
    }
    
    logger.info(f"Profiling results:")
    logger.info(f"  Mean: {stats['mean_ms']:.2f} ms")
    logger.info(f"  Std: {stats['std_ms']:.2f} ms")
    logger.info(f"  Min: {stats['min_ms']:.2f} ms")
    logger.info(f"  Max: {stats['max_ms']:.2f} ms")
    logger.info(f"  Throughput: {stats['throughput_fps']:.1f} FPS")
    
    return stats


def analyze_torchscript_graph(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze TorchScript model graph
    
    Args:
        model_path: Path to TorchScript model
        
    Returns:
        Graph analysis results
    """
    # Load model
    model = torch.jit.load(str(model_path))
    
    # Get graph
    graph = model.graph
    
    # Count operations
    op_counts = {}
    total_params = 0
    
    for node in graph.nodes():
        op_type = node.kind()
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # Count parameters
        for output in node.outputs():
            if output.type().kind() == 'TensorType':
                shape = output.type().sizes()
                if shape:
                    params = 1
                    for dim in shape:
                        if dim:
                            params *= dim
                    total_params += params
    
    # Get inputs and outputs
    inputs = []
    for input in graph.inputs():
        if input.type().kind() == 'TensorType':
            inputs.append({
                'name': input.debugName(),
                'shape': input.type().sizes(),
                'dtype': input.type().dtype()
            })
    
    outputs = []
    for output in graph.outputs():
        if output.type().kind() == 'TensorType':
            outputs.append({
                'name': output.debugName(),
                'shape': output.type().sizes(),
                'dtype': output.type().dtype()
            })
    
    analysis = {
        'total_nodes': len(list(graph.nodes())),
        'op_counts': op_counts,
        'estimated_params': total_params,
        'inputs': inputs,
        'outputs': outputs,
        'file_size_mb': Path(model_path).stat().st_size / 1024 / 1024
    }
    
    return analysis