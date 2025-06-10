"""
Model Exporter
=============

Main module for exporting trained models to various formats.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
import json
import torch
import torch.nn as nn

from .onnx import ONNXExporter
from .torchscript import TorchScriptExporter
from .inference import create_inference_script

logger = logging.getLogger(__name__)

# Try to import TensorRT
try:
    from .tensorrt import TensorRTExporter
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available")


class ExportFormat(Enum):
    """Supported export formats"""
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    ALL = "all"


@dataclass
class ExportConfig:
    """Configuration for model export"""
    
    # Export settings
    formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.PYTORCH])
    output_dir: Path = Path("exports")
    model_name: str = "model"
    
    # Input configuration
    input_shape: Tuple[int, ...] = (3, 224, 224)
    batch_size: int = 1
    dynamic_batch: bool = True
    
    # Optimization settings
    optimize: bool = True
    quantize: bool = False
    fp16: bool = False
    
    # ONNX specific
    onnx_opset: int = 14
    onnx_simplify: bool = True
    
    # TorchScript specific
    torchscript_trace: bool = True  # Use tracing instead of scripting
    
    # TensorRT specific
    trt_fp16: bool = True
    trt_int8: bool = False
    trt_workspace_size: int = 1 << 30  # 1GB
    trt_max_batch_size: int = 32
    
    # Metadata
    include_metadata: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    verify_outputs: bool = True
    tolerance: float = 1e-5
    
    # Additional files
    create_inference_script: bool = True
    create_requirements: bool = True
    create_readme: bool = True


class ModelExporter:
    """
    Main class for exporting models to various formats
    """
    
    def __init__(self, config: ExportConfig):
        """
        Initialize exporter
        
        Args:
            config: Export configuration
        """
        self.config = config
        self.exported_files = {}
        
        # Create output directory
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exporters
        self.exporters = {
            ExportFormat.TORCHSCRIPT: TorchScriptExporter(config),
            ExportFormat.ONNX: ONNXExporter(config)
        }
        
        if TENSORRT_AVAILABLE:
            self.exporters[ExportFormat.TENSORRT] = TensorRTExporter(config)
    
    def export(
        self,
        model: nn.Module,
        checkpoint_path: Optional[Union[str, Path]] = None,
        example_input: Optional[torch.Tensor] = None
    ) -> Dict[str, Path]:
        """
        Export model to specified formats
        
        Args:
            model: Model to export
            checkpoint_path: Optional checkpoint to load
            example_input: Example input tensor
            
        Returns:
            Dictionary of format to exported file path
        """
        logger.info(f"Starting model export to {self.config.output_dir}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(model, checkpoint_path)
        
        # Prepare model
        model.eval()
        device = next(model.parameters()).device
        
        # Create example input if not provided
        if example_input is None:
            example_input = self._create_example_input(device)
        
        # Export to each format
        if ExportFormat.ALL in self.config.formats:
            formats = [f for f in ExportFormat if f != ExportFormat.ALL]
        else:
            formats = self.config.formats
        
        for format in formats:
            try:
                if format == ExportFormat.PYTORCH:
                    self._export_pytorch(model)
                else:
                    if format in self.exporters:
                        output_path = self.exporters[format].export(model, example_input)
                        self.exported_files[format] = output_path
                    else:
                        logger.warning(f"Exporter for {format} not available")
                        
            except Exception as e:
                logger.error(f"Failed to export to {format}: {e}")
                if hasattr(self.config, 'raise_on_error') and self.config.raise_on_error:
                    raise
        
        # Verify exports
        if self.config.verify_outputs:
            self._verify_exports(model, example_input)
        
        # Create additional files
        if self.config.create_inference_script:
            self._create_inference_script()
        
        if self.config.create_requirements:
            self._create_requirements()
        
        if self.config.create_readme:
            self._create_readme()
        
        # Save metadata
        if self.config.include_metadata:
            self._save_metadata()
        
        logger.info(f"Export completed. Files saved to {self.config.output_dir}")
        
        return self.exported_files
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        
        # Extract metadata if available
        if isinstance(checkpoint, dict):
            self.config.metadata.update({
                'checkpoint_path': str(checkpoint_path),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_metric': checkpoint.get('best_metric', 'unknown')
            })
    
    def _export_pytorch(self, model: nn.Module) -> Path:
        """Export PyTorch model with metadata"""
        output_path = self.config.output_dir / f"{self.config.model_name}_pytorch.pth"
        
        logger.info(f"Exporting PyTorch model to {output_path}")
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': self.config.metadata.get('model_config', {}),
            'export_config': {
                'input_shape': self.config.input_shape,
                'batch_size': self.config.batch_size,
                'dynamic_batch': self.config.dynamic_batch
            },
            'metadata': self.config.metadata
        }
        
        # Add model architecture if possible
        if hasattr(model, 'config'):
            checkpoint['model_config'] = model.config
        
        torch.save(checkpoint, output_path)
        
        self.exported_files[ExportFormat.PYTORCH] = output_path
        
        return output_path
    
    def _create_example_input(self, device: torch.device) -> torch.Tensor:
        """Create example input tensor"""
        shape = (self.config.batch_size, *self.config.input_shape)
        return torch.randn(shape, device=device)
    
    def _verify_exports(self, original_model: nn.Module, example_input: torch.Tensor) -> None:
        """Verify exported models produce same outputs"""
        logger.info("Verifying exported models...")
        
        with torch.no_grad():
            original_output = original_model(example_input)
        
        for format, exporter in self.exporters.items():
            if format not in self.exported_files:
                continue
            
            if hasattr(exporter, 'verify'):
                try:
                    is_valid = exporter.verify(
                        self.exported_files[format],
                        example_input,
                        original_output,
                        tolerance=self.config.tolerance
                    )
                    
                    if is_valid:
                        logger.info(f"{format} export verified ✓")
                    else:
                        logger.warning(f"{format} export verification failed")
                        
                except Exception as e:
                    logger.error(f"Failed to verify {format}: {e}")
    
    def _create_inference_script(self) -> None:
        """Create standalone inference script"""
        script_path = self.config.output_dir / "inference.py"
        
        # Get available formats
        available_formats = list(self.exported_files.keys())
        
        script_content = create_inference_script(
            model_name=self.config.model_name,
            input_shape=self.config.input_shape,
            available_formats=available_formats,
            metadata=self.config.metadata
        )
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Created inference script: {script_path}")
    
    def _create_requirements(self) -> None:
        """Create requirements.txt for inference"""
        requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.21.0",
            "Pillow>=9.0.0"
        ]
        
        if ExportFormat.ONNX in self.exported_files:
            requirements.extend([
                "onnx>=1.13.0",
                "onnxruntime>=1.14.0"
            ])
        
        if ExportFormat.TENSORRT in self.exported_files:
            requirements.append("tensorrt>=8.0.0")
        
        req_path = self.config.output_dir / "requirements.txt"
        
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        logger.info(f"Created requirements.txt: {req_path}")
    
    def _create_readme(self) -> None:
        """Create README with usage instructions"""
        readme_content = f"""# {self.config.model_name} - Exported Model

## Available Formats

"""
        for format, path in self.exported_files.items():
            readme_content += f"- **{format.value}**: `{path.name}`\n"
        
        readme_content += f"""
## Usage

### PyTorch

```python
import torch

# Load model
checkpoint = torch.load('{self.config.model_name}_pytorch.pth')
model = YourModelClass()  # Initialize your model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### Inference Script

```bash
python inference.py --image path/to/image.jpg --format pytorch
```

## Model Information

- Input shape: {self.config.input_shape}
- Batch size: {self.config.batch_size}
- Dynamic batch: {self.config.dynamic_batch}

## Requirements

See `requirements.txt` for dependencies.
"""
        
        readme_path = self.config.output_dir / "README.md"
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created README: {readme_path}")
    
    def _save_metadata(self) -> None:
        """Save export metadata"""
        metadata = {
            'export_config': {
                'formats': [f.value for f in self.config.formats],
                'input_shape': self.config.input_shape,
                'batch_size': self.config.batch_size,
                'dynamic_batch': self.config.dynamic_batch,
                'optimize': self.config.optimize,
                'quantize': self.config.quantize
            },
            'exported_files': {
                format.value: str(path.name) 
                for format, path in self.exported_files.items()
            },
            'model_metadata': self.config.metadata
        }
        
        metadata_path = self.config.output_dir / "export_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {metadata_path}")


def export_model(
    model: nn.Module,
    formats: Union[str, List[str]] = "pytorch",
    output_dir: str = "exports",
    **kwargs
) -> Dict[str, Path]:
    """
    Convenience function for model export
    
    Args:
        model: Model to export
        formats: Export format(s)
        output_dir: Output directory
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary of format to exported file path
    """
    # Parse formats
    if isinstance(formats, str):
        if formats == "all":
            format_list = [ExportFormat.ALL]
        else:
            format_list = [ExportFormat(formats)]
    else:
        format_list = [ExportFormat(f) for f in formats]
    
    # Create config
    config = ExportConfig(
        formats=format_list,
        output_dir=Path(output_dir),
        **kwargs
    )
    
    # Export
    exporter = ModelExporter(config)
    return exporter.export(model)