"""
ONNX Export
===========

Export models to ONNX format for interoperability.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import ONNX dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX dependencies not available")

try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False


class ONNXExporter:
    """
    Export PyTorch models to ONNX format
    """
    
    def __init__(self, config: Any):
        """
        Initialize ONNX exporter
        
        Args:
            config: Export configuration
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX and onnxruntime required for ONNX export")
        
        self.config = config
    
    def export(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export model to ONNX format
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            output_path: Output file path
            
        Returns:
            Path to exported ONNX model
        """
        if output_path is None:
            output_path = self.config.output_dir / f"{self.config.model_name}.onnx"
        
        logger.info(f"Exporting to ONNX: {output_path}")
        
        # Prepare model
        model.eval()
        
        # Determine input/output names
        input_names = ['input']
        output_names = ['output']
        
        # Dynamic axes for batch size
        dynamic_axes = None
        if self.config.dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_input,
                str(output_path),
                export_params=True,
                opset_version=self.config.onnx_opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        # Verify the exported model
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            raise
        
        # Simplify if requested
        if self.config.onnx_simplify and ONNXSIM_AVAILABLE:
            output_path = self._simplify_model(output_path)
        
        # Optimize if requested
        if self.config.optimize:
            output_path = self._optimize_model(output_path)
        
        logger.info(f"ONNX export completed: {output_path}")
        
        return output_path
    
    def _simplify_model(self, model_path: Path) -> Path:
        """Simplify ONNX model using onnx-simplifier"""
        logger.info("Simplifying ONNX model...")
        
        try:
            model = onnx.load(str(model_path))
            model_simp, check = simplify(model)
            
            if check:
                # Save simplified model
                output_path = model_path.parent / f"{model_path.stem}_simplified.onnx"
                onnx.save(model_simp, str(output_path))
                logger.info(f"Model simplified successfully: {output_path}")
                return output_path
            else:
                logger.warning("Simplification check failed, using original model")
                return model_path
                
        except Exception as e:
            logger.warning(f"Failed to simplify model: {e}")
            return model_path
    
    def _optimize_model(self, model_path: Path) -> Path:
        """Optimize ONNX model for inference"""
        return optimize_onnx_model(
            model_path,
            optimization_level=self.config.get('onnx_optimization_level', 'all')
        )
    
    def verify(
        self,
        model_path: Path,
        example_input: torch.Tensor,
        expected_output: torch.Tensor,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Verify ONNX model outputs match PyTorch
        
        Args:
            model_path: Path to ONNX model
            example_input: Example input tensor
            expected_output: Expected output from PyTorch model
            tolerance: Numerical tolerance
            
        Returns:
            True if outputs match within tolerance
        """
        try:
            # Create ONNX runtime session
            session = ort.InferenceSession(str(model_path))
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            onnx_input = {input_name: example_input.cpu().numpy()}
            
            # Run inference
            onnx_output = session.run(None, onnx_input)[0]
            
            # Convert to tensor for comparison
            onnx_output_tensor = torch.from_numpy(onnx_output)
            
            # Compare outputs
            is_close = torch.allclose(
                expected_output.cpu(),
                onnx_output_tensor,
                rtol=tolerance,
                atol=tolerance
            )
            
            if is_close:
                max_diff = torch.max(torch.abs(expected_output.cpu() - onnx_output_tensor))
                logger.info(f"ONNX verification passed. Max difference: {max_diff:.6f}")
            else:
                logger.warning("ONNX verification failed. Outputs do not match.")
            
            return is_close
            
        except Exception as e:
            logger.error(f"ONNX verification error: {e}")
            return False


def optimize_onnx_model(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    optimization_level: str = 'all',
    custom_ops: Optional[List[str]] = None
) -> Path:
    """
    Optimize ONNX model for inference
    
    Args:
        model_path: Path to ONNX model
        output_path: Output path for optimized model
        optimization_level: Optimization level ('basic', 'extended', 'all')
        custom_ops: List of custom operators to preserve
        
    Returns:
        Path to optimized model
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not available, skipping optimization")
        return Path(model_path)
    
    model_path = Path(model_path)
    
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Optimizing ONNX model: {model_path}")
    
    try:
        # Available optimizations
        optimizations = []
        
        if optimization_level in ['basic', 'all']:
            optimizations.extend([
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_batch_norm',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
            ])
        
        if optimization_level in ['extended', 'all']:
            optimizations.extend([
                'fuse_add_bias_into_conv',
                'fuse_conv_bn',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ])
        
        # Load model
        model = onnx.load(str(model_path))
        
        # Apply optimizations
        from onnx import optimizer
        optimized_model = optimizer.optimize(model, optimizations)
        
        # Save optimized model
        onnx.save(optimized_model, str(output_path))
        
        # Verify optimized model
        onnx.checker.check_model(optimized_model)
        
        # Report size reduction
        original_size = model_path.stat().st_size / 1024 / 1024  # MB
        optimized_size = output_path.stat().st_size / 1024 / 1024  # MB
        reduction = (1 - optimized_size / original_size) * 100
        
        logger.info(
            f"Model optimized: {original_size:.1f}MB → {optimized_size:.1f}MB "
            f"({reduction:.1f}% reduction)"
        )
        
        return output_path
        
    except Exception as e:
        logger.error(f"ONNX optimization failed: {e}")
        return model_path


def verify_onnx_model(
    model_path: Union[str, Path],
    test_input_shape: Optional[Tuple[int, ...]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify and analyze ONNX model
    
    Args:
        model_path: Path to ONNX model
        test_input_shape: Shape for test input
        verbose: Whether to print detailed info
        
    Returns:
        Dictionary with model information
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX required for model verification")
    
    model_path = Path(model_path)
    
    # Load model
    model = onnx.load(str(model_path))
    
    # Basic verification
    try:
        onnx.checker.check_model(model)
        is_valid = True
        error_msg = None
    except Exception as e:
        is_valid = False
        error_msg = str(e)
    
    # Extract model info
    info = {
        'is_valid': is_valid,
        'error': error_msg,
        'file_size_mb': model_path.stat().st_size / 1024 / 1024,
        'ir_version': model.ir_version,
        'producer_name': model.producer_name,
        'producer_version': model.producer_version,
        'opset_version': model.opset_import[0].version if model.opset_import else None,
        'graph_name': model.graph.name,
        'inputs': [],
        'outputs': []
    }
    
    # Input/output information
    for input in model.graph.input:
        input_info = {
            'name': input.name,
            'shape': [d.dim_value if d.dim_value > 0 else d.dim_param 
                     for d in input.type.tensor_type.shape.dim],
            'dtype': input.type.tensor_type.elem_type
        }
        info['inputs'].append(input_info)
    
    for output in model.graph.output:
        output_info = {
            'name': output.name,
            'shape': [d.dim_value if d.dim_value > 0 else d.dim_param 
                     for d in output.type.tensor_type.shape.dim],
            'dtype': output.type.tensor_type.elem_type
        }
        info['outputs'].append(output_info)
    
    # Count operations
    op_counts = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    info['op_counts'] = op_counts
    info['total_ops'] = len(model.graph.node)
    
    # Test inference if requested
    if test_input_shape and ONNX_AVAILABLE:
        try:
            session = ort.InferenceSession(str(model_path))
            
            # Create test input
            input_name = session.get_inputs()[0].name
            test_input = torch.randn(test_input_shape).numpy()
            
            # Run inference
            import time
            start_time = time.time()
            output = session.run(None, {input_name: test_input})
            inference_time = time.time() - start_time
            
            info['test_inference'] = {
                'success': True,
                'inference_time_ms': inference_time * 1000,
                'output_shape': output[0].shape
            }
        except Exception as e:
            info['test_inference'] = {
                'success': False,
                'error': str(e)
            }
    
    if verbose:
        logger.info(f"ONNX Model Information for {model_path.name}:")
        logger.info(f"  Valid: {info['is_valid']}")
        logger.info(f"  Size: {info['file_size_mb']:.1f} MB")
        logger.info(f"  Opset: {info['opset_version']}")
        logger.info(f"  Operations: {info['total_ops']}")
        logger.info(f"  Inputs: {info['inputs']}")
        logger.info(f"  Outputs: {info['outputs']}")
    
    return info