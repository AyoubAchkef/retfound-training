"""
Base Callback Classes
====================

Defines the base callback interface and callback handler for training hooks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Abstract base class for callbacks
    
    Callbacks provide hooks into the training process at various stages.
    """
    
    def __init__(self):
        """Initialize callback"""
        self.trainer = None
        self.model = None
        self.config = None
    
    def set_trainer(self, trainer):
        """
        Set trainer reference
        
        Args:
            trainer: Trainer instance
        """
        self.trainer = trainer
        self.model = trainer.model
        self.config = trainer.config
    
    def on_train_begin(self, trainer):
        """
        Called at the beginning of training
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_train_end(self, trainer):
        """
        Called at the end of training
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        """
        Called at the beginning of each epoch
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
        """
        pass
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """
        Called at the end of each epoch
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        pass
    
    def on_batch_begin(self, trainer, batch_idx: int):
        """
        Called at the beginning of each batch
        
        Args:
            trainer: Trainer instance
            batch_idx: Current batch index
        """
        pass
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float,
        outputs: Any,
        targets: Any
    ):
        """
        Called at the end of each batch
        
        Args:
            trainer: Trainer instance
            batch_idx: Current batch index
            loss: Batch loss
            outputs: Model outputs
            targets: Batch targets
        """
        pass
    
    def on_validation_begin(self, trainer):
        """
        Called at the beginning of validation
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_validation_end(
        self,
        trainer,
        val_metrics: Dict[str, float]
    ):
        """
        Called at the end of validation
        
        Args:
            trainer: Trainer instance
            val_metrics: Validation metrics
        """
        pass
    
    def on_checkpoint_save(
        self,
        trainer,
        checkpoint: Dict[str, Any],
        filepath: str
    ):
        """
        Called when saving a checkpoint
        
        Args:
            trainer: Trainer instance
            checkpoint: Checkpoint dictionary
            filepath: Save path
        """
        pass
    
    def on_checkpoint_load(
        self,
        trainer,
        checkpoint: Dict[str, Any],
        filepath: str
    ):
        """
        Called when loading a checkpoint
        
        Args:
            trainer: Trainer instance
            checkpoint: Checkpoint dictionary
            filepath: Load path
        """
        pass
    
    def on_exception(self, trainer, exception: Exception):
        """
        Called when an exception occurs during training
        
        Args:
            trainer: Trainer instance
            exception: Exception that occurred
        """
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get callback state for checkpointing
        
        Returns:
            State dictionary
        """
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load callback state from checkpoint
        
        Args:
            state_dict: State dictionary
        """
        pass


class CallbackHandler:
    """
    Manages multiple callbacks and calls their hooks
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback handler
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
        self.stop_training = False
    
    def add_callback(self, callback: Callback):
        """
        Add a callback
        
        Args:
            callback: Callback to add
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callback):
        """
        Remove a callback
        
        Args:
            callback: Callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def set_trainer(self, trainer):
        """
        Set trainer reference for all callbacks
        
        Args:
            trainer: Trainer instance
        """
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, trainer):
        """Call on_train_begin for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_train_begin(trainer)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_train_begin: {e}")
    
    def on_train_end(self, trainer):
        """Call on_train_end for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_train_end(trainer)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_train_end: {e}")
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Call on_epoch_begin for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_epoch_begin(trainer, epoch)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_epoch_begin: {e}")
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Call on_epoch_end for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(trainer, epoch, train_metrics, val_metrics)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_epoch_end: {e}")
    
    def on_batch_begin(self, trainer, batch_idx: int):
        """Call on_batch_begin for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_batch_begin(trainer, batch_idx)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_batch_begin: {e}")
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float,
        outputs: Any,
        targets: Any
    ):
        """Call on_batch_end for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_batch_end(trainer, batch_idx, loss, outputs, targets)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_batch_end: {e}")
    
    def on_validation_begin(self, trainer):
        """Call on_validation_begin for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_validation_begin(trainer)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_validation_begin: {e}")
    
    def on_validation_end(self, trainer, val_metrics: Dict[str, float]):
        """Call on_validation_end for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_validation_end(trainer, val_metrics)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_validation_end: {e}")
    
    def on_checkpoint_save(
        self,
        trainer,
        checkpoint: Dict[str, Any],
        filepath: str
    ):
        """Call on_checkpoint_save for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_checkpoint_save(trainer, checkpoint, filepath)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_checkpoint_save: {e}")
    
    def on_checkpoint_load(
        self,
        trainer,
        checkpoint: Dict[str, Any],
        filepath: str
    ):
        """Call on_checkpoint_load for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_checkpoint_load(trainer, checkpoint, filepath)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_checkpoint_load: {e}")
    
    def on_exception(self, trainer, exception: Exception):
        """Call on_exception for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_exception(trainer, exception)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_exception: {e}")
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for all callbacks
        
        Returns:
            Dictionary mapping callback names to their states
        """
        state = {}
        for callback in self.callbacks:
            try:
                callback_state = callback.state_dict()
                if callback_state:
                    state[callback.__class__.__name__] = callback_state
            except Exception as e:
                logger.error(f"Error getting state for {callback.__class__.__name__}: {e}")
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state dict for all callbacks
        
        Args:
            state_dict: Dictionary mapping callback names to their states
        """
        for callback in self.callbacks:
            callback_name = callback.__class__.__name__
            if callback_name in state_dict:
                try:
                    callback.load_state_dict(state_dict[callback_name])
                except Exception as e:
                    logger.error(f"Error loading state for {callback_name}: {e}")


class LRSchedulerCallback(Callback):
    """
    Callback for learning rate scheduling
    
    This is a simple wrapper that steps the scheduler after each batch/epoch
    """
    
    def __init__(
        self,
        scheduler,
        step_on: str = 'batch',
        metric: Optional[str] = None
    ):
        """
        Initialize LR scheduler callback
        
        Args:
            scheduler: Learning rate scheduler
            step_on: When to step ('batch' or 'epoch')
            metric: Metric to monitor for ReduceLROnPlateau
        """
        super().__init__()
        self.scheduler = scheduler
        self.step_on = step_on
        self.metric = metric
    
    def on_batch_end(self, trainer, batch_idx, loss, outputs, targets):
        """Step scheduler after batch if configured"""
        if self.step_on == 'batch':
            self.scheduler.step()
    
    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        """Step scheduler after epoch if configured"""
        if self.step_on == 'epoch':
            if self.metric and hasattr(self.scheduler, 'step'):
                # For ReduceLROnPlateau
                metric_value = val_metrics.get(self.metric, 0)
                self.scheduler.step(metric_value)
            else:
                self.scheduler.step()


class EMACallback(Callback):
    """
    Callback for Exponential Moving Average updates
    """
    
    def __init__(self, ema_model):
        """
        Initialize EMA callback
        
        Args:
            ema_model: EMA model instance
        """
        super().__init__()
        self.ema_model = ema_model
    
    def on_batch_end(self, trainer, batch_idx, loss, outputs, targets):
        """Update EMA after each batch"""
        self.ema_model.update()
    
    def on_checkpoint_save(self, trainer, checkpoint, filepath):
        """Add EMA state to checkpoint"""
        checkpoint['ema_state_dict'] = self.ema_model.state_dict()
    
    def on_checkpoint_load(self, trainer, checkpoint, filepath):
        """Load EMA state from checkpoint"""
        if 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
