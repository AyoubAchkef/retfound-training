"""
Utilities Module
===============

Common utilities for the RETFound training system.
"""

from .io import (
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    ensure_dir,
    get_file_size,
    copy_file,
    move_file
)

from .device import (
    get_device,
    get_gpu_info,
    set_device,
    clear_gpu_cache,
    get_optimal_batch_size,
    check_gpu_memory
)

from .logging import (
    setup_logging,
    get_logger,
    log_system_info,
    log_gpu_info,
    log_model_info
)

from .timing import (
    Timer,
    TimeIt,
    format_time,
    ETA,
    Profiler
)

from .reproducibility import (
    set_seed,
    set_deterministic,
    worker_init_fn,
    get_reproducible_dataloader
)

__all__ = [
    # IO utilities
    'load_json',
    'save_json',
    'load_yaml',
    'save_yaml',
    'ensure_dir',
    'get_file_size',
    'copy_file',
    'move_file',
    
    # Device utilities
    'get_device',
    'get_gpu_info',
    'set_device',
    'clear_gpu_cache',
    'get_optimal_batch_size',
    'check_gpu_memory',
    
    # Logging utilities
    'setup_logging',
    'get_logger',
    'log_system_info',
    'log_gpu_info',
    'log_model_info',
    
    # Timing utilities
    'Timer',
    'TimeIt',
    'format_time',
    'ETA',
    'Profiler',
    
    # Reproducibility utilities
    'set_seed',
    'set_deterministic',
    'worker_init_fn',
    'get_reproducible_dataloader'
]