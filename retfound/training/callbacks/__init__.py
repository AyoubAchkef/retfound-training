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
            checkpoint_dir=config.checkpoint_path,
            save_frequency=config.save_frequency,
            save_best=True,
            save_last=True
        )
    )
    
    # Early stopping callback
    callbacks.append(
        EarlyStoppingCallback(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            monitor='val_accuracy',
            mode='max'
        )
    )
    
    # Metrics callback
    callbacks.append(
        MetricsCallback(
            output_dir=config.output_path / 'metrics',
            log_frequency=config.val_frequency
        )
    )
    
    # Logging callbacks
    callbacks.append(
        LoggingCallback(
            log_frequency=config.log_interval
        )
    )
    
    if config.use_tensorboard:
        callbacks.append(
            TensorBoardCallback(
                log_dir=config.output_path / 'tensorboard'
            )
        )
    
    if config.use_wandb:
        callbacks.append(
            WandBCallback(
                project=config.wandb_project,
                entity=config.wandb_entity,
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
