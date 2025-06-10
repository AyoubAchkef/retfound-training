"""
Early Stopping Callback
======================

Implements early stopping to prevent overfitting.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

from .base import Callback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to stop training when metric stops improving
    
    Features:
    - Monitor any metric
    - Configurable patience
    - Min delta for improvement
    - Restore best weights option
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True,
        baseline: Optional[float] = None
    ):
        """
        Initialize early stopping callback
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore best model weights
            verbose: Whether to print messages
            baseline: Baseline value for metric
        """
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.baseline = baseline
        
        # Initialize tracking variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = None
        self.best_weights = None
        self.best_epoch = 0
        
        # Set comparison function
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def on_train_begin(self, trainer):
        """Initialize at training start"""
        self.wait = 0
        self.stopped_epoch = 0
        
        if self.baseline is not None:
            self.best_metric = self.baseline
        else:
            self.best_metric = np.inf if self.mode == 'min' else -np.inf
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Check for improvement at end of epoch"""
        # Get current metric value
        current = None
        if self.monitor in val_metrics:
            current = val_metrics[self.monitor]
        elif self.monitor in train_metrics:
            current = train_metrics[self.monitor]
        else:
            logger.warning(
                f"Early stopping monitor '{self.monitor}' not found in metrics"
            )
            return
        
        # Check if improvement
        if self.monitor_op(current - self.min_delta, self.best_metric):
            # Improvement found
            self.best_metric = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Store best weights
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
            
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} improved to {current:.4f}"
                )
        else:
            # No improvement
            self.wait += 1
            
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve from "
                    f"{self.best_metric:.4f}. Patience: {self.wait}/{self.patience}"
                )
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.callback_handler.stop_training = True
                
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training. Best {self.monitor}: "
                        f"{self.best_metric:.4f} at epoch {self.best_epoch}"
                    )
    
    def on_train_end(self, trainer):
        """Restore best weights if configured"""
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose:
                logger.info(
                    f"Restoring model weights from epoch {self.best_epoch}"
                )
            trainer.model.load_state_dict(self.best_weights)
        
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(
                f"Early stopping triggered at epoch {self.stopped_epoch}"
            )
    
    def state_dict(self) -> Dict[str, Any]:
        """Get callback state"""
        state = {
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
        # Optionally save best weights
        if self.restore_best_weights and self.best_weights is not None:
            state['best_weights'] = self.best_weights
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load callback state"""
        self.wait = state_dict.get('wait', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)
        self.best_metric = state_dict.get('best_metric', self.best_metric)
        self.best_epoch = state_dict.get('best_epoch', 0)
        
        if 'best_weights' in state_dict:
            self.best_weights = state_dict['best_weights']


class ReduceLROnPlateauCallback(Callback):
    """
    Reduce learning rate when metric plateaus
    
    Similar to PyTorch's ReduceLROnPlateau but as a callback
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        mode: str = 'min',
        threshold: float = 0.0001,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        verbose: bool = True
    ):
        """
        Initialize reduce LR on plateau callback
        
        Args:
            monitor: Metric to monitor
            factor: Factor by which to reduce LR
            patience: Number of epochs without improvement before reducing
            min_lr: Minimum learning rate
            mode: 'min' or 'max'
            threshold: Threshold for measuring improvement
            threshold_mode: 'rel' or 'abs'
            cooldown: Epochs to wait after reduction
            verbose: Whether to print messages
        """
        super().__init__()
        
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.verbose = verbose
        
        # Initialize tracking
        self.wait = 0
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        
        # Set comparison function
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - self._get_threshold(b)
        else:
            self.monitor_op = lambda a, b: a > b + self._get_threshold(b)
    
    def _get_threshold(self, value: float) -> float:
        """Calculate threshold based on mode"""
        if self.threshold_mode == 'rel':
            return abs(value * self.threshold)
        else:
            return self.threshold
    
    def on_train_begin(self, trainer):
        """Initialize at training start"""
        self.wait = 0
        self.cooldown_counter = 0
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Check if LR should be reduced"""
        # Get current metric
        current = None
        if self.monitor in val_metrics:
            current = val_metrics[self.monitor]
        elif self.monitor in train_metrics:
            current = train_metrics[self.monitor]
        else:
            return
        
        # Check if in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            return
        
        # Check if improvement
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                # Reduce learning rate
                old_lrs = []
                new_lrs = []
                
                for param_group in trainer.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                    old_lrs.append(old_lr)
                    new_lrs.append(new_lr)
                
                if self.verbose:
                    logger.info(
                        f"ReduceLROnPlateau: reducing learning rate. "
                        f"New LRs: {new_lrs}"
                    )
                
                self.cooldown_counter = self.cooldown
                self.wait = 0
                self.num_bad_epochs += 1
