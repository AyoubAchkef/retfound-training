"""
Model export utilities for RETFound.
Supports exporting models trained on v6.1 dataset with proper metadata.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.jit import script

from ..core.constants import DATASET_V61_CLASSES, DATASET_V40_CLASSES
from ..core.exceptions import ExportError
from ..models import load_model
from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RETFoundExporter:
    """Export RETFound models to various formats."""
    
    def __init__(
        self,
        checkpoint_path: str,
        output_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize model exporter.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_dir: Directory to save exported models
            device: Device to use for export
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir) if output_dir else self.checkpoint_path.parent / "export"
        self.device = device or get_device()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and metadata
        self.model, self.config, self.metadata = self._load_model_and_metadata()
        
        # Detect dataset version
        self.dataset_version = self._detect_dataset_version()
        
        logger.info(f"Initialized exporter for {self.dataset_version} model")
    
    def _load_model_and_metadata(self) -> tuple:
        """Load model, config, and metadata from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Extract config
            config = checkpoint.get('config', {})
            
            # Extract metadata
            metadata = checkpoint.get('metadata', {})
            
            # Load model
            model = load_model(
                config=config,
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            
            # Ensure model is in eval mode
            model.eval()
            
            return model, config, metadata
            
        except Exception as e:
            raise ExportError(f"Failed to load model: {str(e)}")
    
    def _detect_dataset_version(self) -> str:
        """Detect dataset version from model configuration."""
        # Check metadata first
        if 'dataset_version' in self.metadata:
            return self.metadata['dataset_version']
        
        # Check config
        if 'dataset_version' in self.config.get('data', {}):
            return self.config['data']['dataset_version']
        
        # Infer from number of classes
        num_classes = getattr(self.model, 'num_classes', None)
        if num_classes == 28:
            logger.info("Detected 28 classes, assuming v6.1 dataset")
            return "v6.1"
        elif num_classes == 22:
            logger.info("Detected 22 classes, assuming v4.0 dataset")
            return "v4.0"
        else:
            logger.warning("Could not detect dataset version, defaulting to v6.1")
            return "v6.1"
    
    def export(
        self,
        formats: Union[str, List[str]] = ["onnx", "torchscript"],
        optimize: bool = True,
        quantize: Optional[str] = None,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 11,
        include_metadata: bool = True
    ) -> Dict[str, Path]:
        """
        Export model to specified formats.
        
        Args:
            formats: Export format(s) - "onnx", "torchscript", "tensorrt"
            optimize: Whether to optimize the exported model
            quantize: Quantization mode - "int8", "fp16", or None
            input_shape: Input tensor shape for export
            opset_version: ONNX opset version
            include_metadata: Whether to include metadata in export
            
        Returns:
            Dictionary mapping format to exported file path
        """
        if isinstance(formats, str):
            formats = [formats]
        
        exported_paths = {}
        
        for format_name in formats:
            logger.info(f"Exporting to {format_name} format...")
            
            if format_name.lower() == "onnx":
                path = self._export_onnx(
                    input_shape=input_shape,
                    opset_version=opset_version,
                    optimize=optimize,
                    quantize=quantize,
                    include_metadata=include_metadata
                )
            elif format_name.lower() == "torchscript":
                path = self._export_torchscript(
                    input_shape=input_shape,
                    optimize=optimize,
                    quantize=quantize,
                    include_metadata=include_metadata
                )
            elif format_name.lower() == "tensorrt":
                path = self._export_tensorrt(
                    input_shape=input_shape,
                    optimize=optimize,
                    precision=quantize or "fp32",
                    include_metadata=include_metadata
                )
            else:
                logger.warning(f"Unsupported format: {format_name}")
                continue
            
            exported_paths[format_name] = path
            logger.info(f"Successfully exported to {path}")
        
        # Save metadata separately
        if include_metadata:
            self._save_metadata(exported_paths)
        
        return exported_paths
    
    def _export_onnx(
        self,
        input_shape: tuple,
        opset_version: int,
        optimize: bool,
        quantize: Optional[str],
        include_metadata: bool
    ) -> Path:
        """Export model to ONNX format."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ExportError("ONNX export requires onnx and onnxruntime packages")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Define output path
        output_path = self.output_dir / "model.onnx"
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                do_constant_folding=optimize,
                export_params=True,
                verbose=False
            )
        
        # Add metadata to ONNX model
        if include_metadata:
            onnx_model = onnx.load(str(output_path))
            
            # Add metadata
            metadata_props = []
            
            # Dataset version and classes
            metadata_props.append(
                onnx.StringStringEntryProto(key='dataset_version', value=self.dataset_version)
            )
            metadata_props.append(
                onnx.StringStringEntryProto(key='num_classes', value=str(self.model.num_classes))
            )
            
            # Class names
            if self.dataset_version == "v6.1":
                class_names = DATASET_V61_CLASSES
            else:
                class_names = DATASET_V40_CLASSES
            
            metadata_props.append(
                onnx.StringStringEntryProto(key='class_names', value=json.dumps(class_names))
            )
            
            # Model metadata
            if self.metadata:
                for key, value in self.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_props.append(
                            onnx.StringStringEntryProto(key=f'model_{key}', value=str(value))
                        )
            
            # Add to model
            onnx_model.metadata_props.extend(metadata_props)
            
            # Save updated model
            onnx.save(onnx_model, str(output_path))
        
        # Optimize if requested
        if optimize:
            self._optimize_onnx(output_path)
        
        # Quantize if requested
        if quantize:
            output_path = self._quantize_onnx(output_path, quantize)
        
        # Validate exported model
        self._validate_onnx(output_path, input_shape)
        
        return output_path
    
    def _export_torchscript(
        self,
        input_shape: tuple,
        optimize: bool,
        quantize: Optional[str],
        include_metadata: bool
    ) -> Path:
        """Export model to TorchScript format."""
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Define output path
        output_path = self.output_dir / "model.pt"
        
        # Convert to TorchScript
        with torch.no_grad():
            # Try to script first, fall back to trace
            try:
                scripted_model = torch.jit.script(self.model)
            except:
                logger.warning("Scripting failed, falling back to tracing")
                scripted_model = torch.jit.trace(self.model, dummy_input)
            
            # Optimize if requested
            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            # Add metadata as attributes
            if include_metadata:
                # Dataset info
                scripted_model.dataset_version = self.dataset_version
                scripted_model.num_classes = self.model.num_classes
                
                # Class names
                if self.dataset_version == "v6.1":
                    scripted_model.class_names = DATASET_V61_CLASSES
                else:
                    scripted_model.class_names = DATASET_V40_CLASSES
                
                # Additional metadata
                if hasattr(self.model, 'modality'):
                    scripted_model.modality = self.model.modality
            
            # Quantize if requested
            if quantize:
                if quantize == "int8":
                    scripted_model = torch.quantization.quantize_dynamic(
                        scripted_model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                elif quantize == "fp16":
                    scripted_model = scripted_model.half()
            
            # Save model
            torch.jit.save(scripted_model, str(output_path))
        
        # Validate exported model
        self._validate_torchscript(output_path, input_shape)
        
        return output_path
    
    def _export_tensorrt(
        self,
        input_shape: tuple,
        optimize: bool,
        precision: str,
        include_metadata: bool
    ) -> Path:
        """Export model to TensorRT format."""
        try:
            import tensorrt as trt
            import torch2trt
        except ImportError:
            raise ExportError("TensorRT export requires tensorrt and torch2trt packages")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Define output path
        output_path = self.output_dir / "model.trt"
        
        # Convert precision string to torch2trt format
        precision_map = {
            "fp32": None,
            "fp16": torch.float16,
            "int8": torch.int8
        }
        trt_precision = precision_map.get(precision, None)
        
        # Convert to TensorRT
        with torch.no_grad():
            trt_model = torch2trt.torch2trt(
                self.model,
                [dummy_input],
                fp16_mode=(trt_precision == torch.float16),
                int8_mode=(trt_precision == torch.int8),
                optimize=optimize,
                max_workspace_size=1 << 30  # 1GB
            )
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(trt_model.engine.serialize())
        
        # Save metadata separately for TensorRT
        if include_metadata:
            metadata_path = output_path.with_suffix('.json')
            metadata = {
                'dataset_version': self.dataset_version,
                'num_classes': self.model.num_classes,
                'class_names': DATASET_V61_CLASSES if self.dataset_version == "v6.1" else DATASET_V40_CLASSES,
                'input_shape': list(input_shape),
                'precision': precision,
                'model_metadata': self.metadata
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return output_path
    
    def _optimize_onnx(self, model_path: Path):
        """Optimize ONNX model."""
        try:
            from onnxruntime.transformers import optimizer
        except ImportError:
            logger.warning("ONNX optimization requires onnxruntime.transformers")
            return
        
        # Run optimization
        optimized_path = model_path.with_name("model_optimized.onnx")
        optimizer.optimize_model(
            str(model_path),
            str(optimized_path),
            optimization_level=99
        )
        
        # Replace original with optimized
        optimized_path.replace(model_path)
    
    def _quantize_onnx(self, model_path: Path, quantize: str) -> Path:
        """Quantize ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            logger.warning("ONNX quantization requires onnxruntime.quantization")
            return model_path
        
        # Define quantized path
        quantized_path = model_path.with_name(f"model_{quantize}.onnx")
        
        # Quantize model
        if quantize == "int8":
            quantize_dynamic(
                str(model_path),
                str(quantized_path),
                weight_type=QuantType.QInt8
            )
        else:
            logger.warning(f"Unsupported quantization type: {quantize}")
            return model_path
        
        return quantized_path
    
    def _validate_onnx(self, model_path: Path, input_shape: tuple):
        """Validate exported ONNX model."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            logger.warning("Skipping ONNX validation (requires onnxruntime)")
            return
        
        # Check model
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = ort.InferenceSession(str(model_path))
        
        # Create test input
        test_input = torch.randn(input_shape).numpy()
        
        # Run inference
        input_name = ort_session.get_inputs()[0].name
        output = ort_session.run(None, {input_name: test_input})
        
        # Validate output shape
        expected_output_shape = (input_shape[0], self.model.num_classes)
        actual_output_shape = output[0].shape
        
        if actual_output_shape != expected_output_shape:
            raise ExportError(
                f"Output shape mismatch: expected {expected_output_shape}, "
                f"got {actual_output_shape}"
            )
        
        logger.info("ONNX model validation passed")
    
    def _validate_torchscript(self, model_path: Path, input_shape: tuple):
        """Validate exported TorchScript model."""
        # Load model
        loaded_model = torch.jit.load(str(model_path))
        loaded_model.eval()
        
        # Create test input
        test_input = torch.randn(input_shape)
        
        # Run inference
        with torch.no_grad():
            output = loaded_model(test_input)
        
        # Validate output shape
        expected_output_shape = (input_shape[0], self.model.num_classes)
        actual_output_shape = tuple(output.shape)
        
        if actual_output_shape != expected_output_shape:
            raise ExportError(
                f"Output shape mismatch: expected {expected_output_shape}, "
                f"got {actual_output_shape}"
            )
        
        # Check metadata
        if hasattr(loaded_model, 'dataset_version'):
            logger.info(f"Model metadata - Dataset: {loaded_model.dataset_version}, "
                       f"Classes: {loaded_model.num_classes}")
        
        logger.info("TorchScript model validation passed")
    
    def _save_metadata(self, exported_paths: Dict[str, Path]):
        """Save comprehensive metadata file."""
        metadata_path = self.output_dir / "export_metadata.json"
        
        # Compile metadata
        export_metadata = {
            'export_info': {
                'timestamp': str(torch.datetime.datetime.now()),
                'source_checkpoint': str(self.checkpoint_path),
                'exported_formats': list(exported_paths.keys()),
                'exported_files': {fmt: str(path) for fmt, path in exported_paths.items()}
            },
            'model_info': {
                'dataset_version': self.dataset_version,
                'num_classes': self.model.num_classes,
                'class_names': DATASET_V61_CLASSES if self.dataset_version == "v6.1" else DATASET_V40_CLASSES,
                'model_type': self.config.get('model', {}).get('type', 'unknown'),
                'input_size': self.config.get('data', {}).get('input_size', 224)
            },
            'training_info': self.metadata,
            'inference_info': {
                'preprocessing': {
                    'resize': 224,
                    'normalize': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                },
                'postprocessing': {
                    'activation': 'softmax',
                    'temperature': getattr(self.model, 'temperature', 1.0)
                }
            }
        }
        
        # Add v6.1 specific info
        if self.dataset_version == "v6.1":
            export_metadata['model_info']['modality_info'] = {
                'fundus_classes': list(range(18)),
                'oct_classes': list(range(18, 28)),
                'fundus_class_names': DATASET_V61_CLASSES[:18],
                'oct_class_names': DATASET_V61_CLASSES[18:]
            }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(export_metadata, f, indent=2)
        
        logger.info(f"Saved export metadata to {metadata_path}")