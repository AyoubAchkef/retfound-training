"""
Model Export Module
==================

Export trained models to various formats for deployment.
"""

from .exporter import (
    ModelExporter,
    ExportConfig,
    ExportFormat,
    export_model
)

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
