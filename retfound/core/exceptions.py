"""Custom exceptions for RETFound Training Framework."""

from typing import Optional, Any


class RETFoundError(Exception):
    """Base exception for all RETFound errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ConfigurationError(RETFoundError):
    """Raised when there's an error in configuration."""
    pass


class DatasetError(RETFoundError):
    """Raised when there's an error with dataset loading or processing."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset is not found."""
    pass


class DataCorruptedError(DatasetError):
    """Raised when dataset data is corrupted."""
    pass


class ModelError(RETFoundError):
    """Raised when there's an error with model architecture or weights."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class WeightsNotFoundError(ModelError):
    """Raised when model weights are not found."""
    pass


class TrainingError(RETFoundError):
    """Raised when there's an error during training."""
    pass


class ExportError(RETFoundError):
    """Raised when there's an error during model export."""
    pass


class ValidationError(RETFoundError):
    """Raised when validation fails."""
    pass


class CheckpointError(RETFoundError):
    """Raised when there's an error with checkpoint loading/saving."""
    pass


class DependencyError(RETFoundError):
    """Raised when a required dependency is missing."""
    pass


class HardwareError(RETFoundError):
    """Raised when hardware requirements are not met."""
    pass


class InferenceError(RETFoundError):
    """Raised when there's an error during inference."""
    pass


class EvaluationError(RETFoundError):
    """Raised when there's an error during model evaluation."""
    pass


# Utility functions for better error handling
def handle_import_error(module_name: str, package_name: Optional[str] = None) -> None:
    """Handle import errors with helpful messages."""
    if package_name is None:
        package_name = module_name
    
    raise DependencyError(
        f"Required module '{module_name}' not found.",
        f"Please install it with: pip install {package_name}"
    )


def check_gpu_memory(required_gb: float = 16.0) -> None:
    """Check if GPU has enough memory."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < required_gb:
                raise HardwareError(
                    f"Insufficient GPU memory: {gpu_memory_gb:.1f}GB available, "
                    f"{required_gb:.1f}GB required.",
                    "Consider reducing batch size or enabling gradient checkpointing."
                )
    except ImportError:
        raise DependencyError("PyTorch not installed")


def validate_file_exists(file_path: Any, file_type: str = "file") -> None:
    """Validate that a file exists."""
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"{file_type} not found: {path}")
    
    if file_type == "directory" and not path.is_dir():
        raise ValidationError(f"Expected directory but found file: {path}")
    
    if file_type == "file" and not path.is_file():
        raise ValidationError(f"Expected file but found directory: {path}")
