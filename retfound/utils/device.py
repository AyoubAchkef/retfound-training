"""
Device Utilities
===============

GPU/CPU device management utilities.
"""

import logging
import subprocess
from typing import Optional, Dict, List, Tuple
import torch

logger = logging.getLogger(__name__)


def get_device(
    device: Optional[str] = None,
    gpu_id: Optional[int] = None
) -> torch.device:
    """
    Get torch device
    
    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)
        gpu_id: GPU ID to use (overrides device string)
        
    Returns:
        torch.device object
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            logger.warning("CUDA not available, using CPU")
    
    if gpu_id is not None and torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
    
    return torch.device(device)


def get_gpu_info() -> Dict[int, Dict[str, any]]:
    """
    Get information about available GPUs
    
    Returns:
        Dictionary mapping GPU ID to info dict
    """
    if not torch.cuda.is_available():
        return {}
    
    gpu_info = {}
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        
        gpu_info[i] = {
            'name': props.name,
            'total_memory_gb': props.total_memory / 1024**3,
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count,
            'is_available': True
        }
        
        # Try to get current memory usage
        try:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            gpu_info[i]['allocated_memory_gb'] = allocated
            gpu_info[i]['reserved_memory_gb'] = reserved
            gpu_info[i]['free_memory_gb'] = props.total_memory / 1024**3 - allocated
        except:
            pass
    
    return gpu_info


def set_device(device: Optional[int] = None) -> None:
    """
    Set CUDA device
    
    Args:
        device: Device ID to set as current
    """
    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)
        logger.info(f"Set CUDA device to {device}")


def clear_gpu_cache() -> None:
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("Cleared GPU cache")


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    starting_batch_size: int = 32,
    device: Optional[torch.device] = None,
    safety_margin: float = 0.9
) -> int:
    """
    Find optimal batch size for model and GPU
    
    Args:
        model: Model to test
        input_shape: Input shape (without batch dimension)
        starting_batch_size: Initial batch size to try
        device: Device to test on
        safety_margin: Memory safety margin (0-1)
        
    Returns:
        Optimal batch size
    """
    if device is None:
        device = next(model.parameters()).device
    
    if device.type != 'cuda':
        logger.warning("Optimal batch size detection only works on GPU")
        return starting_batch_size
    
    model.eval()
    batch_size = starting_batch_size
    
    # Binary search for optimal batch size
    min_batch = 1
    max_batch = starting_batch_size * 4
    optimal_batch = min_batch
    
    while min_batch <= max_batch:
        batch_size = (min_batch + max_batch) // 2
        
        try:
            # Clear cache
            clear_gpu_cache()
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Check memory usage
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            if allocated / total < safety_margin:
                optimal_batch = batch_size
                min_batch = batch_size + 1
            else:
                max_batch = batch_size - 1
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                max_batch = batch_size - 1
                clear_gpu_cache()
            else:
                raise e
    
    logger.info(f"Optimal batch size: {optimal_batch}")
    return optimal_batch


def check_gpu_memory(
    device: Optional[torch.device] = None,
    threshold: float = 0.9
) -> Tuple[bool, float]:
    """
    Check if GPU memory usage is below threshold
    
    Args:
        device: Device to check
        threshold: Memory usage threshold (0-1)
        
    Returns:
        Tuple of (is_below_threshold, current_usage_ratio)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        return True, 0.0
    
    allocated = torch.cuda.memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    usage_ratio = allocated / total
    
    return usage_ratio < threshold, usage_ratio


def get_nvidia_smi_info() -> Optional[str]:
    """
    Get nvidia-smi output
    
    Returns:
        nvidia-smi output string or None if not available
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def monitor_gpu_memory(
    func: callable,
    device: Optional[torch.device] = None,
    interval: float = 0.1
) -> Dict[str, float]:
    """
    Monitor GPU memory usage during function execution
    
    Args:
        func: Function to monitor
        device: Device to monitor
        interval: Monitoring interval in seconds
        
    Returns:
        Dictionary with memory statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        result = func()
        return {'max_allocated_gb': 0, 'max_reserved_gb': 0}
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    # Run function
    result = func()
    
    # Get peak memory usage
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    
    return {
        'max_allocated_gb': max_allocated,
        'max_reserved_gb': max_reserved,
        'result': result
    }


def auto_select_device(
    preferred_gpu_memory_gb: float = 10.0,
    excluded_gpus: Optional[List[int]] = None
) -> torch.device:
    """
    Automatically select best available device
    
    Args:
        preferred_gpu_memory_gb: Minimum GPU memory required
        excluded_gpus: List of GPU IDs to exclude
        
    Returns:
        Selected device
    """
    if not torch.cuda.is_available():
        logger.info("No GPU available, using CPU")
        return torch.device('cpu')
    
    excluded_gpus = excluded_gpus or []
    gpu_info = get_gpu_info()
    
    # Find GPUs with enough memory
    suitable_gpus = []
    for gpu_id, info in gpu_info.items():
        if gpu_id in excluded_gpus:
            continue
        
        free_memory = info.get('free_memory_gb', info['total_memory_gb'])
        if free_memory >= preferred_gpu_memory_gb:
            suitable_gpus.append((gpu_id, free_memory))
    
    if not suitable_gpus:
        # Use GPU with most free memory
        gpu_id = max(gpu_info.keys(), 
                    key=lambda x: gpu_info[x].get('free_memory_gb', 0))
        logger.warning(
            f"No GPU with {preferred_gpu_memory_gb}GB free memory found. "
            f"Using GPU {gpu_id} with {gpu_info[gpu_id].get('free_memory_gb', 0):.1f}GB free"
        )
    else:
        # Use GPU with most free memory among suitable ones
        gpu_id = max(suitable_gpus, key=lambda x: x[1])[0]
        logger.info(f"Selected GPU {gpu_id} with {suitable_gpus[0][1]:.1f}GB free memory")
    
    return torch.device(f'cuda:{gpu_id}')