"""
Learning Rate Schedulers for RETFound
====================================

Implements various learning rate scheduling strategies including
warmup, cosine annealing, and OneCycleLR.
"""

import math
import logging
from typing import Optional, List, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..core.config import RETFoundConfig
from ..core.registry import Registry

logger = logging.getLogger(__name__)

# Scheduler registry
SCHEDULER_REGISTRY = Registry("schedulers")


def register_scheduler(name: str):
    """Decorator to register a scheduler"""
    def decorator(cls):
        SCHEDULER_REGISTRY.register(name, cls)
        return cls
    return decorator


class BaseScheduler(_LRScheduler):
    """Base class for custom schedulers"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        super().__init__(optimizer, last_epoch, verbose)


@register_scheduler("linear_warmup_cosine")
class LinearWarmupCosineAnnealing(BaseScheduler):
    """
    Linear warmup followed by cosine annealing
    
    Learning rate starts from warmup_lr, linearly increases to base_lr
    over warmup_epochs, then follows cosine annealing to min_lr.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        warmup_lr: float,
        min_lr: float,
        max_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize scheduler
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            warmup_lr: Starting learning rate for warmup
            min_lr: Minimum learning rate
            max_epochs: Total number of epochs
            last_epoch: Last epoch index
            verbose: Whether to print LR updates
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_progress = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]


@register_scheduler("cosine_warmup")
class CosineAnnealingWithWarmup(BaseScheduler):
    """
    Cosine annealing with optional warmup
    
    Similar to LinearWarmupCosineAnnealing but with cosine-based warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        min_lr: float,
        max_steps: int,
        warmup_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize scheduler
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            min_lr: Minimum learning rate
            max_steps: Total number of steps
            warmup_lr: Starting learning rate for warmup
            last_epoch: Last epoch index
            verbose: Whether to print LR updates
        """
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr or min_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Cosine warmup
            warmup_progress = self.last_epoch / self.warmup_steps
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - warmup_progress)))
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]


@register_scheduler("onecycle_warmup")
class OneCycleLRWithWarmup(torch.optim.lr_scheduler.OneCycleLR):
    """
    OneCycleLR with extended warmup options
    
    Extends PyTorch's OneCycleLR with additional warmup configurations
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: int,
        warmup_steps: Optional[int] = None,
        warmup_lr: Optional[float] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.85,
        max_momentum: Union[float, List[float]] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        three_phase: bool = False,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize OneCycleLR with warmup
        
        Args:
            optimizer: Optimizer to schedule
            max_lr: Maximum learning rate(s)
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps (overrides pct_start if set)
            warmup_lr: Starting LR for warmup (overrides div_factor if set)
            pct_start: Percentage of cycle for increasing LR
            anneal_strategy: Annealing strategy ('cos' or 'linear')
            cycle_momentum: Whether to cycle momentum
            base_momentum: Base momentum value(s)
            max_momentum: Maximum momentum value(s)
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = initial_lr / final_div_factor
            three_phase: Whether to use three-phase schedule
            last_epoch: Last epoch index
            verbose: Whether to print LR updates
        """
        # Adjust parameters based on warmup settings
        if warmup_steps is not None:
            pct_start = warmup_steps / total_steps
        
        if warmup_lr is not None:
            # Calculate div_factor from warmup_lr
            if isinstance(max_lr, list):
                div_factor = max_lr[0] / warmup_lr
            else:
                div_factor = max_lr / warmup_lr
        
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
            verbose=verbose
        )


@register_scheduler("exponential")
class ExponentialLRWithWarmup(BaseScheduler):
    """Exponential decay with optional warmup"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        warmup_epochs: int = 0,
        warmup_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize scheduler
        
        Args:
            optimizer: Optimizer to schedule
            gamma: Decay factor
            warmup_epochs: Number of warmup epochs
            warmup_lr: Starting learning rate for warmup
            last_epoch: Last epoch index
            verbose: Whether to print LR updates
        """
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr or 0.0
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_progress = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]
        else:
            # Exponential decay
            return [
                base_lr * self.gamma ** (self.last_epoch - self.warmup_epochs)
                for base_lr in self.base_lrs
            ]


@register_scheduler("polynomial")
class PolynomialLRWithWarmup(BaseScheduler):
    """Polynomial decay with optional warmup"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
        warmup_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize scheduler
        
        Args:
            optimizer: Optimizer to schedule
            max_epochs: Total number of epochs
            power: Polynomial power
            min_lr: Minimum learning rate
            warmup_epochs: Number of warmup epochs
            warmup_lr: Starting learning rate for warmup
            last_epoch: Last epoch index
            verbose: Whether to print LR updates
        """
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr or min_lr
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch"""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_progress = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            factor = (1 - progress) ** self.power
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: Optimizer,
    config: RETFoundConfig,
    scheduler_name: Optional[str] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler based on configuration
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object
        scheduler_name: Specific scheduler to use
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    # Determine scheduler name
    if scheduler_name is None:
        # Default to OneCycleLR
        scheduler_name = "onecycle"
    
    # Handle different scheduler types
    if scheduler_name == "onecycle":
        if num_training_steps is None:
            logger.warning(
                "OneCycleLR requires num_training_steps, "
                "will be set later"
            )
            return None
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.base_lr,
            total_steps=num_training_steps,
            pct_start=config.warmup_epochs / config.epochs if config.warmup_epochs > 0 else 0.1,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            **kwargs
        )
    
    elif scheduler_name == "linear_warmup_cosine":
        scheduler = LinearWarmupCosineAnnealing(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            warmup_lr=config.warmup_lr,
            min_lr=config.min_lr,
            max_epochs=config.epochs,
            **kwargs
        )
    
    elif scheduler_name == "cosine_warmup":
        if num_training_steps is None:
            num_training_steps = config.epochs
        
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_steps=config.warmup_epochs,
            min_lr=config.min_lr,
            max_steps=num_training_steps,
            warmup_lr=config.warmup_lr,
            **kwargs
        )
    
    elif scheduler_name == "exponential":
        scheduler = ExponentialLRWithWarmup(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
            warmup_epochs=config.warmup_epochs,
            warmup_lr=config.warmup_lr,
            **kwargs
        )
    
    elif scheduler_name == "polynomial":
        scheduler = PolynomialLRWithWarmup(
            optimizer,
            max_epochs=config.epochs,
            power=kwargs.get('power', 1.0),
            min_lr=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_lr=config.warmup_lr,
            **kwargs
        )
    
    elif scheduler_name in SCHEDULER_REGISTRY:
        # Custom scheduler from registry
        scheduler_class = SCHEDULER_REGISTRY.get(scheduler_name)
        scheduler = scheduler_class(optimizer, config=config, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Available schedulers: onecycle, linear_warmup_cosine, "
            f"cosine_warmup, exponential, polynomial, "
            f"{list(SCHEDULER_REGISTRY.keys())}"
        )
    
    logger.info(f"Created {scheduler_name} scheduler")
    
    return scheduler


def update_scheduler_steps(
    scheduler: _LRScheduler,
    num_training_steps: int
) -> _LRScheduler:
    """
    Update scheduler with correct number of training steps
    
    This is useful for schedulers like OneCycleLR that need to know
    the total number of steps upfront.
    
    Args:
        scheduler: Existing scheduler or None
        num_training_steps: Total number of training steps
        
    Returns:
        Updated scheduler
    """
    if scheduler is None:
        return None
    
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        # OneCycleLR needs to be recreated with correct steps
        optimizer = scheduler.optimizer
        
        # Extract configuration from existing scheduler
        max_lr = scheduler.max_lrs
        pct_start = scheduler.total_steps * scheduler.pct_start / num_training_steps
        
        # Create new scheduler
        new_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=pct_start,
            anneal_strategy=scheduler.anneal_strategy,
            cycle_momentum=scheduler.cycle_momentum,
            base_momentum=scheduler.base_momentums,
            max_momentum=scheduler.max_momentums,
            div_factor=scheduler.div_factor,
            final_div_factor=scheduler.final_div_factor,
            three_phase=scheduler.three_phase,
            last_epoch=scheduler.last_epoch,
            verbose=scheduler.verbose
        )
        
        logger.info(f"Updated OneCycleLR with {num_training_steps} steps")
        
        return new_scheduler
    
    return scheduler
