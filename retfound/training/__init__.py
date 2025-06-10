"""
Training Package for RETFound
============================

Handles all training-related operations including trainers, optimizers,
schedulers, losses, and callbacks.
"""

from .trainer import (
    RETFoundTrainer,
    BaseTrainer,
    create_trainer
)

from .optimizers import (
    SAM,
    create_optimizer,
    get_parameter_groups,
    EMA
)

from .schedulers import (
    create_scheduler,
    OneCycleLRWithWarmup,
    CosineAnnealingWithWarmup,
    LinearWarmupCosineAnnealing
)

from .losses import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    AsymmetricLoss,
    create_loss_function,
    MixupCutmixCriterion
)

from .callbacks import (
    Callback,
    CallbackHandler,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    MetricsCallback,
    LRSchedulerCallback,
    EMACallback,
    create_callbacks
)

__all__ = [
    # Trainer
    'RETFoundTrainer',
    'BaseTrainer',
    'create_trainer',
    
    # Optimizers
    'SAM',
    'create_optimizer',
    'get_parameter_groups',
    'EMA',
    
    # Schedulers
    'create_scheduler',
    'OneCycleLRWithWarmup',
    'CosineAnnealingWithWarmup',
    'LinearWarmupCosineAnnealing',
    
    # Losses
    'LabelSmoothingCrossEntropy',
    'FocalLoss',
    'AsymmetricLoss',
    'create_loss_function',
    'MixupCutmixCriterion',
    
    # Callbacks
    'Callback',
    'CallbackHandler',
    'CheckpointCallback',
    'EarlyStoppingCallback',
    'LoggingCallback',
    'MetricsCallback',
    'LRSchedulerCallback',
    'EMACallback',
    'create_callbacks'
]
