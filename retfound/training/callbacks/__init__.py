"""
Callbacks Package for RETFound Training
======================================

Implements various callbacks for monitoring and controlling the training process.
"""

from .base import Callback, CallbackHandler
from .checkpoint import CheckpointCallback
from .early_stopping import EarlyStoppingCallback
from .logging import LoggingCallback, TensorBoardCallback, WandBCallback
from .metrics import MetricsCallback

# Create placeholder classes for missing callbacks
class LRSchedulerCallback(Callback):
    """Placeholder for LR Scheduler callback"""
    def __init__(self, scheduler=None, **kwargs):
        super().__init__()
        self.scheduler = scheduler

class EMACallback(Callback):
    """Placeholder for EMA callback"""
    def __init__(self, model=None, **kwargs):
        super().__init__()
        self.model = model

class VisualizationCallback(Callback):
    """Placeholder for Visualization callback"""
    def __init__(self, **kwargs):
        super().__init__()

class WandbCallback(Callback):
    """Placeholder for Wandb callback"""
    def __init__(self, **kwargs):
        super().__init__()

from typing import List
from ...core.config import RETFoundConfig


def create_callbacks(
    config: RETFoundConfig,
    trainer=None
) -> List[Callback]:
    """
    Create callbacks based on configuration
    
    Args:
        config: Configuration object
        trainer: Trainer instance
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Checkpoint callback
    callbacks.append(
        CheckpointCallback(
            checkpoint_dir=getattr(config, 'checkpoint_path', 'checkpoints'),
            save_frequency=getattr(config, 'save_frequency', 5),
            save_best=True,
            save_last=True
        )
    )
    
    # Early stopping callback
    callbacks.append(
        EarlyStoppingCallback(
            patience=getattr(config, 'early_stopping_patience', 10),
            min_delta=getattr(config, 'early_stopping_min_delta', 0.0001),
            monitor='val_accuracy',
            mode='max'
        )
    )
    
    # Metrics callback
    callbacks.append(
        MetricsCallback(
            output_dir=getattr(config, 'output_path', 'outputs') / 'metrics',
            log_frequency=getattr(config, 'val_frequency', 1)
        )
    )
    
    # Logging callbacks
    callbacks.append(
        LoggingCallback(
            log_frequency=getattr(config, 'log_interval', 10)
        )
    )
    
    if getattr(config, 'use_tensorboard', False):
        callbacks.append(
            TensorBoardCallback(
                log_dir=getattr(config, 'output_path', 'outputs') / 'tensorboard'
            )
        )
    
    if getattr(config, 'use_wandb', False):
        callbacks.append(
            WandBCallback(
                project=getattr(config, 'wandb_project', 'retfound'),
                entity=getattr(config, 'wandb_entity', None),
                config=config
            )
        )
    
    return callbacks


__all__ = [
    'Callback',
    'CallbackHandler',
    'CheckpointCallback',
    'EarlyStoppingCallback',
    'LoggingCallback',
    'TensorBoardCallback',
    'WandBCallback',
    'MetricsCallback',
    'LRSchedulerCallback',
    'EMACallback',
    'VisualizationCallback',
    'WandbCallback',
    'create_callbacks'
]
