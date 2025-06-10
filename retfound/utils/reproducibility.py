"""
Reproducibility Utilities
========================

Utilities for ensuring reproducible training results.
"""

import random
import logging
import os
from typing import Optional, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic algorithms
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        set_deterministic(True)
    
    logger.info(f"Random seed set to {seed}")


def set_deterministic(mode: bool = True) -> None:
    """
    Enable/disable deterministic algorithms in PyTorch
    
    Args:
        mode: Whether to enable deterministic mode
        
    Note:
        Deterministic mode may impact performance
    """
    if mode:
        # PyTorch deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # cuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For some operations that don't have deterministic implementations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger.info("Deterministic mode enabled (may impact performance)")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
            del os.environ['CUBLAS_WORKSPACE_CONFIG']
        
        logger.info("Deterministic mode disabled")


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker process with unique random seed
    
    Args:
        worker_id: Worker process ID
        
    Note:
        Use this with DataLoader for reproducible data loading
    """
    # Get initial seed (set by set_seed)
    initial_seed = torch.initial_seed() % 2**32
    
    # Create unique seed for each worker
    worker_seed = initial_seed + worker_id
    
    # Set seeds
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    # Also set for any libraries that might use time-based seeds
    import time
    time.sleep(0.001 * worker_id)


def get_reproducible_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with reproducible behavior
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        seed: Random seed (uses global seed if None)
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with reproducible behavior
    """
    # Create generator with fixed seed
    if seed is None:
        seed = torch.initial_seed() % 2**32
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Create DataLoader with worker init function
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator if shuffle else None,
        **kwargs
    )


class ReproducibilityManager:
    """
    Context manager for reproducible experiments
    """
    
    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = True,
        warn_only: bool = True
    ):
        """
        Initialize reproducibility manager
        
        Args:
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
            warn_only: Whether to only warn on non-deterministic operations
        """
        self.seed = seed
        self.deterministic = deterministic
        self.warn_only = warn_only
        
        # Store previous states
        self.prev_random_state = None
        self.prev_np_state = None
        self.prev_torch_state = None
        self.prev_cuda_state = None
        self.prev_cudnn_deterministic = None
        self.prev_cudnn_benchmark = None
        self.prev_deterministic_algorithms = None
    
    def __enter__(self):
        """Enter reproducible context"""
        # Save current states
        self.prev_random_state = random.getstate()
        self.prev_np_state = np.random.get_state()
        self.prev_torch_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self.prev_cuda_state = torch.cuda.get_rng_state_all()
        
        self.prev_cudnn_deterministic = torch.backends.cudnn.deterministic
        self.prev_cudnn_benchmark = torch.backends.cudnn.benchmark
        
        # Set reproducible state
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit reproducible context and restore previous state"""
        # Restore previous states
        random.setstate(self.prev_random_state)
        np.random.set_state(self.prev_np_state)
        torch.set_rng_state(self.prev_torch_state)
        
        if torch.cuda.is_available() and self.prev_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.prev_cuda_state)
        
        torch.backends.cudnn.deterministic = self.prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = self.prev_cudnn_benchmark
        
        # Reset deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)


def check_reproducibility(
    func: callable,
    n_runs: int = 3,
    seed: int = 42,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """
    Check if a function produces reproducible results
    
    Args:
        func: Function to check
        n_runs: Number of runs to compare
        seed: Random seed to use
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if results are reproducible
    """
    results = []
    
    for i in range(n_runs):
        with ReproducibilityManager(seed=seed):
            result = func()
            results.append(result)
    
    # Compare results
    reproducible = True
    
    for i in range(1, n_runs):
        if isinstance(results[0], torch.Tensor):
            if not torch.allclose(results[0], results[i], rtol=rtol, atol=atol):
                reproducible = False
                break
        elif isinstance(results[0], np.ndarray):
            if not np.allclose(results[0], results[i], rtol=rtol, atol=atol):
                reproducible = False
                break
        else:
            if results[0] != results[i]:
                reproducible = False
                break
    
    if reproducible:
        logger.info(f"Function '{func.__name__}' is reproducible")
    else:
        logger.warning(f"Function '{func.__name__}' is NOT reproducible")
    
    return reproducible


def log_random_states() -> Dict[str, Any]:
    """
    Log current random states for debugging
    
    Returns:
        Dictionary of current random states
    """
    states = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    
    # Log some sample random numbers
    logger.debug("Current random states:")
    logger.debug(f"  Python random: {random.random()}")
    logger.debug(f"  NumPy random: {np.random.rand()}")
    logger.debug(f"  PyTorch random: {torch.rand(1).item()}")
    
    if torch.cuda.is_available():
        logger.debug(f"  CUDA random: {torch.cuda.FloatTensor(1).uniform_().item()}")
    
    return states


def save_random_states(filepath: str) -> None:
    """
    Save current random states to file
    
    Args:
        filepath: Path to save states
    """
    states = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        states['cuda'] = torch.cuda.get_rng_state_all()
    
    torch.save(states, filepath)
    logger.info(f"Random states saved to {filepath}")


def load_random_states(filepath: str) -> None:
    """
    Load random states from file
    
    Args:
        filepath: Path to load states from
    """
    states = torch.load(filepath)
    
    random.setstate(states['python_random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    
    if torch.cuda.is_available() and 'cuda' in states:
        torch.cuda.set_rng_state_all(states['cuda'])
    
    logger.info(f"Random states loaded from {filepath}")


# Utility functions for common reproducibility issues

def make_reproducible_optimizer(optimizer_class: type, **kwargs) -> Any:
    """
    Create optimizer with reproducible behavior
    
    Args:
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        **kwargs: Optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Some optimizers have non-deterministic behavior with certain settings
    if optimizer_class.__name__ == 'Adam' and 'amsgrad' not in kwargs:
        kwargs['amsgrad'] = True  # More stable
    
    return optimizer_class(**kwargs)


def check_deterministic_operations() -> None:
    """Check for non-deterministic operations in the model"""
    # This will raise errors for non-deterministic operations
    torch.use_deterministic_algorithms(True, warn_only=False)
    
    logger.info("Checking for non-deterministic operations...")
    
    # Common non-deterministic operations to check
    operations = [
        ("Upsampling", lambda: torch.nn.functional.interpolate(
            torch.randn(1, 1, 4, 4), scale_factor=2
        )),
        ("Index add", lambda: torch.zeros(5, 3).index_add_(
            0, torch.tensor([0, 2]), torch.randn(2, 3)
        )),
        ("Scatter add", lambda: torch.zeros(3, 5).scatter_add_(
            0, torch.tensor([[0, 1, 2, 0, 0]]), torch.randn(1, 5)
        )),
    ]
    
    for name, op in operations:
        try:
            op()
            logger.info(f"  {name}: Deterministic ✓")
        except RuntimeError as e:
            logger.warning(f"  {name}: Non-deterministic ✗")
            logger.warning(f"    {str(e)}")
    
    # Reset to warn only
    torch.use_deterministic_algorithms(True, warn_only=True)