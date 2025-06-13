"""
Model Export Module
==================

Export trained models to various formats for deployment.
"""

from .exporter import (
    RETFoundExporter,
)

# Create alias for backward compatibility
ModelExporter = RETFoundExporter

# Create placeholder classes/functions if they don't exist
try:
    from .exporter import ExportConfig
except ImportError:
    # Create a simple config class
    class ExportConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    from .exporter import ExportFormat
except ImportError:
    # Create a simple enum-like class
    class ExportFormat:
        ONNX = "onnx"
        TORCHSCRIPT = "torchscript"
        TENSORRT = "tensorrt"

try:
    from .exporter import export_model
except ImportError:
    # Create a simple export function
    def export_model(model, output_path, format="onnx", **kwargs):
        """Simple export function using RETFoundExporter"""
        exporter = RETFoundExporter(model, **kwargs)
        if format.lower() == "onnx":
            return exporter.export_onnx(output_path)
        elif format.lower() == "torchscript":
            return exporter.export_torchscript(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

from .onnx import (
    ONNXExporter,
    optimize_onnx_model,
    verify_onnx_model
)

from .torchscript import (
    TorchScriptExporter,
    trace_model,
    script_model,
    optimize_torchscript
)

from .inference import (
    InferenceModel,
    RETFoundPredictor,
    create_inference_script,
    benchmark_inference
)

# Optional TensorRT support
try:
    from .tensorrt import (
        TensorRTExporter,
        create_trt_engine,
        optimize_trt_engine
    )
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    TensorRTExporter = None

__all__ = [
    # Main exporter
    'ModelExporter',
    'ExportConfig',
    'ExportFormat',
    'export_model',
    
    # ONNX
    'ONNXExporter',
    'optimize_onnx_model',
    'verify_onnx_model',
    
    # TorchScript
    'TorchScriptExporter',
    'trace_model',
    'script_model',
    'optimize_torchscript',
    
    # TensorRT
    'TensorRTExporter',
    'create_trt_engine',
    'optimize_trt_engine',
    
    # Inference
    'InferenceModel',
    'RETFoundPredictor',
    'create_inference_script',
    'benchmark_inference',
    
    # Availability flags
    'TENSORRT_AVAILABLE'
]
