"""
TensorRT Export
===============

Export models to TensorRT for optimized NVIDIA GPU inference.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import TensorRT dependencies
try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
    
    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT dependencies not available")


class TensorRTExporter:
    """
    Export PyTorch models to TensorRT format
    """
    
    def __init__(self, config: Any):
        """
        Initialize TensorRT exporter
        
        Args:
            config: Export configuration
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT and torch_tensorrt required for TensorRT export")
        
        self.config = config
    
    def export(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export model to TensorRT format
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            output_path: Output file path
            
        Returns:
            Path to exported TensorRT model
        """
        if output_path is None:
            output_path = self.config.output_dir / f"{self.config.model_name}_trt.ts"
        
        logger.info(f"Exporting to TensorRT: {output_path}")
        
        # Prepare model
        model.eval()
        device = example_input.device
        
        # Configure TensorRT compilation
        compile_spec = {
            "inputs": [
                torch_tensorrt.Input(
                    shape=example_input.shape,
                    dtype=example_input.dtype
                )
            ],
            "enabled_precisions": set()
        }
        
        # Add precision modes
        compile_spec["enabled_precisions"].add(torch.float32)
        
        if self.config.trt_fp16:
            compile_spec["enabled_precisions"].add(torch.float16)
            logger.info("Enabled FP16 precision")
        
        if self.config.trt_int8:
            compile_spec["enabled_precisions"].add(torch.int8)
            logger.info("Enabled INT8 precision (requires calibration)")
        
        # Additional settings
        compile_spec["workspace_size"] = self.config.trt_workspace_size
        compile_spec["max_batch_size"] = self.config.trt_max_batch_size
        
        # Compile model
        try:
            trt_model = torch_tensorrt.compile(model, **compile_spec)
            
            # Save compiled model
            torch.jit.save(trt_model, str(output_path))
            
            logger.info(f"TensorRT export completed: {output_path}")
            
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            # Try alternative export method
            output_path = self._export_via_onnx(model, example_input, output_path)
        
        return output_path
    
    def _export_via_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: Path
    ) -> Path:
        """
        Export to TensorRT via ONNX (alternative method)
        """
        logger.info("Falling back to ONNX->TensorRT conversion")
        
        # First export to ONNX
        onnx_path = output_path.parent / f"{output_path.stem}.onnx"
        
        torch.onnx.export(
            model,
            example_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Convert ONNX to TensorRT
        engine_path = output_path.with_suffix('.engine')
        engine = self._build_engine_from_onnx(onnx_path, engine_path)
        
        if engine:
            # Clean up ONNX file
            onnx_path.unlink()
            return engine_path
        else:
            raise RuntimeError("Failed to build TensorRT engine")
    
    def _build_engine_from_onnx(
        self,
        onnx_path: Path,
        engine_path: Path
    ) -> Optional[Any]:
        """
        Build TensorRT engine from ONNX model
        """
        import onnx
        
        # Create builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.trt_workspace_size
        
        # Set precision
        if self.config.trt_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        if self.config.trt_int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here
        
        # Build engine
        logger.info("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        
        if engine:
            # Serialize and save
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            logger.info(f"TensorRT engine saved to {engine_path}")
            return engine
        else:
            logger.error("Failed to build TensorRT engine")
            return None
    
    def verify(
        self,
        model_path: Path,
        example_input: torch.Tensor,
        expected_output: torch.Tensor,
        tolerance: float = 1e-3
    ) -> bool:
        """
        Verify TensorRT model outputs
        
        Args:
            model_path: Path to TensorRT model
            example_input: Example input tensor
            expected_output: Expected output
            tolerance: Numerical tolerance
            
        Returns:
            True if outputs match
        """
        try:
            if model_path.suffix == '.ts':
                # TorchScript with TensorRT
                trt_model = torch.jit.load(str(model_path))
                trt_model.eval()
                
                with torch.no_grad():
                    trt_output = trt_model(example_input)
                
            elif model_path.suffix == '.engine':
                # Pure TensorRT engine
                trt_output = self._run_trt_engine(model_path, example_input)
                trt_output = torch.from_numpy(trt_output).to(expected_output.device)
            
            else:
                logger.error(f"Unknown TensorRT format: {model_path.suffix}")
                return False
            
            # Compare outputs
            is_close = torch.allclose(
                expected_output,
                trt_output,
                rtol=tolerance,
                atol=tolerance
            )
            
            if is_close:
                max_diff = torch.max(torch.abs(expected_output - trt_output))
                logger.info(f"TensorRT verification passed. Max difference: {max_diff:.6f}")
            else:
                logger.warning("TensorRT verification failed. Outputs do not match.")
            
            return is_close
            
        except Exception as e:
            logger.error(f"TensorRT verification error: {e}")
            return False
    
    def _run_trt_engine(
        self,
        engine_path: Path,
        input_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Run inference with TensorRT engine
        """
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings = [], [], []
        stream = torch.cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = np.empty(size, dtype=dtype)
            device_mem = torch.cuda.ByteTensor(host_mem.nbytes)
            
            # Append to the appropriate list
            bindings.append(int(device_mem.data_ptr()))
            
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        # Transfer input data
        inputs[0]['host'] = input_tensor.cpu().numpy().ravel()
        
        # Transfer input to device
        with stream:
            for inp in inputs:
                inp['device'].copy_(torch.from_numpy(inp['host']).cuda())
            
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
            
            # Transfer predictions back
            for out in outputs:
                torch.cuda.memcpy_async(
                    out['host'], out['device'],
                    out['host'].nbytes,
                    torch.cuda.memcpyDeviceToHost,
                    stream
                )
            
            stream.synchronize()
        
        return outputs[0]['host']


def create_trt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 32,
    workspace_size: int = 1 << 30,
    calibrator: Optional[Any] = None
) -> bool:
    """
    Create TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision
        max_batch_size: Maximum batch size
        workspace_size: GPU memory for optimization
        calibrator: INT8 calibrator (required if int8=True)
        
    Returns:
        True if successful
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT not available")
        return False
    
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    logger.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parsing error: {parser.get_error(error)}")
            return False
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    # Optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    
    # Get input shape from network
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    
    # Set dynamic batch dimension
    min_shape = [1] + list(input_shape[1:])
    opt_shape = [max_batch_size // 2] + list(input_shape[1:])
    max_shape = [max_batch_size] + list(input_shape[1:])
    
    profile.set_shape(
        input_tensor.name,
        min_shape,
        opt_shape,
        max_shape
    )
    config.add_optimization_profile(profile)
    
    # Set precision
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 precision")
    
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator
        logger.info("Enabled INT8 precision")
    
    # Build engine
    logger.info("Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    
    if engine:
        # Serialize and save
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        logger.info(f"TensorRT engine saved to {engine_path}")
        return True
    else:
        logger.error("Failed to build TensorRT engine")
        return False


def optimize_trt_engine(
    engine_path: Union[str, Path],
    optimization_level: int = 3
) -> None:
    """
    Apply additional optimizations to TensorRT engine
    
    Args:
        engine_path: Path to TensorRT engine
        optimization_level: Optimization level (1-5)
    """
    # TensorRT engines are already optimized during building
    # This function is a placeholder for future optimizations
    logger.info(f"TensorRT engine at {engine_path} is already optimized")


def benchmark_trt_engine(
    engine_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark TensorRT engine performance
    
    Args:
        engine_path: Path to TensorRT engine
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Benchmark statistics
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT not available")
        return {}
    
    import time
    
    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs, outputs, bindings = [], [], []
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = np.empty(size, dtype=dtype)
        device_mem = torch.cuda.ByteTensor(host_mem.nbytes)
        
        bindings.append(int(device_mem.data_ptr()))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32).ravel()
    
    # Warmup
    logger.info(f"Warming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        inputs[0]['host'] = dummy_input
        inputs[0]['device'].copy_(torch.from_numpy(inputs[0]['host']).cuda())
        context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"Benchmarking with {num_runs} runs...")
    times = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        inputs[0]['host'] = dummy_input
        inputs[0]['device'].copy_(torch.from_numpy(inputs[0]['host']).cuda())
        context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'throughput_fps': 1000 / np.mean(times)
    }