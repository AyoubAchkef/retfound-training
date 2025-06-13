"""
Logging Utilities
================

Logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import platform
import torch

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    colorize: bool = True
) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        colorize: Whether to colorize console output
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if colorize and sys.stdout.isatty():
        try:
            import colorlog
            color_format = colorlog.ColoredFormatter(
                '%(log_color)s' + format_string,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_format)
        except ImportError:
            console_handler.setFormatter(logging.Formatter(format_string))
    else:
        console_handler.setFormatter(logging.Formatter(format_string))
    
    handlers.append(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    # Set levels for common libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Get logger with optional custom level
    
    Args:
        name: Logger name
        level: Optional logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    
    return logger


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system information
    
    Args:
        logger: Logger to use (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("=" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 70)
    
    # Platform info
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # CPU info
    logger.info(f"CPU: {platform.processor()}")
    
    if PSUTIL_AVAILABLE:
        logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        
        # Memory info
        mem = psutil.virtual_memory()
        logger.info(f"RAM: {mem.total / 1024**3:.1f} GB total, {mem.available / 1024**3:.1f} GB available")
    else:
        logger.info("CPU/Memory details: psutil not available")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        logger.info(f"CUDA Available: No")
    
    logger.info("=" * 70)


def log_gpu_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log detailed GPU information
    
    Args:
        logger: Logger to use
    """
    if logger is None:
        logger = logging.getLogger()
    
    if not torch.cuda.is_available():
        logger.info("No GPU available")
        return
    
    logger.info("GPU INFORMATION")
    logger.info("-" * 50)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        
        logger.info(f"\nGPU {i}: {props.name}")
        logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        logger.info(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  Multiprocessors: {props.multi_processor_count}")
        logger.info(f"  CUDA Cores: ~{props.multi_processor_count * 64}")  # Approximate
        
        # Current memory usage
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"  Memory Allocated: {allocated:.1f} GB")
            logger.info(f"  Memory Reserved: {reserved:.1f} GB")


def log_model_info(
    model: torch.nn.Module,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log model information
    
    Args:
        model: PyTorch model
        logger: Logger to use
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("MODEL INFORMATION")
    logger.info("-" * 50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    logger.info(f"Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params/1e6:.1f}M)")
    
    # Model size estimate
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    logger.info(f"Model size: {total_size / 1024**2:.1f} MB")
    
    # Layer counts
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    logger.info("\nLayer counts:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:  # Only show layers that appear multiple times
            logger.info(f"  {layer_type}: {count}")


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that plays nicely with tqdm progress bars
    """
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # Fallback to print if tqdm not available
            print(self.format(record))


def setup_file_rotation(
    log_dir: Union[str, Path],
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.handlers.RotatingFileHandler:
    """
    Setup rotating file handler
    
    Args:
        log_dir: Directory for log files
        max_bytes: Maximum size per log file
        backup_count: Number of backup files to keep
        
    Returns:
        Rotating file handler
    """
    from logging.handlers import RotatingFileHandler
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"retfound_{datetime.now().strftime('%Y%m%d')}.log"
    
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    return handler
