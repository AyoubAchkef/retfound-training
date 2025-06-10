"""
Training Strategies
==================

Different training strategies for single GPU, multi-GPU, etc.
"""

from typing import Dict, Type

from .base import TrainingStrategy
from .single_gpu import SingleGPUStrategy

# Import DDP only if available
try:
    from .ddp import DDPStrategy
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False
    DDPStrategy = None

# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[TrainingStrategy]] = {
    'single_gpu': SingleGPUStrategy,
}

if DDP_AVAILABLE:
    STRATEGY_REGISTRY['ddp'] = DDPStrategy
    STRATEGY_REGISTRY['multi_gpu'] = DDPStrategy  # Alias


def get_strategy(name: str) -> Type[TrainingStrategy]:
    """
    Get training strategy by name
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy class
        
    Raises:
        ValueError: If strategy not found
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: {name}. Available: {available}"
        )
    
    return STRATEGY_REGISTRY[name]


def register_strategy(name: str, strategy_class: Type[TrainingStrategy]):
    """
    Register a custom training strategy
    
    Args:
        name: Strategy name
        strategy_class: Strategy class
    """
    STRATEGY_REGISTRY[name] = strategy_class


__all__ = [
    'TrainingStrategy',
    'SingleGPUStrategy',
    'DDPStrategy',
    'get_strategy',
    'register_strategy',
    'STRATEGY_REGISTRY'
]