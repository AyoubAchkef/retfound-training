"""
Trainer Classes for RETFound
===========================

Implements the main training logic with support for advanced features
like mixed precision, gradient accumulation, and callback system.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from collections import defaultdict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.config import RETFoundConfig
from ..core.registry import Registry
from ..metrics import OphthalmologyMetrics
from ..data.transforms import MixupCutmixTransform
from .optimizers import create_optimizer, EMA, SAM
from .schedulers import create_scheduler
from .losses import create_loss_function, MixupCutmixCriterion
from .callbacks import CallbackHandler, create_callbacks

logger = logging.getLogger(__name__)

# Trainer registry
TRAINER_REGISTRY = Registry("trainers")


def register_trainer(name: str):
    """Decorator to register a trainer"""
    def decorator(cls):
        TRAINER_REGISTRY.register(name, cls)
        return cls
    return decorator


class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(
        self,
        model: nn.Module,
        config: RETFoundConfig,
        device: Union[str, torch.device] = 'cuda'
    ):
        """
        Initialize base trainer
        
        Args:
            model: Model to train
            config: Configuration object
            device: Device to use for training
        """
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.best_epoch = 0
        
        # Components (to be initialized)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.ema = None
        
        # Metrics
        self.train_metrics = None
        self.val_metrics = None
        
        # History
        self.history = defaultdict(list)
        
        # Callbacks
        self.callback_handler = CallbackHandler()
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.output_dir = self.config.output_path
        self.checkpoint_dir = self.config.checkpoint_path
        
        for directory in [self.output_dir, self.checkpoint_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def setup_training(self, train_dataset=None):
        """Setup training components - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate(self, val_loader: DataLoader):
        """Validate model - must be implemented by subclasses"""
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0
    ) -> Dict[str, List]:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_epoch: Starting epoch
            
        Returns:
            Training history
        """
        # Training start callback
        self.callback_handler.on_train_begin(self)
        
        try:
            for epoch in range(start_epoch, self.config.epochs):
                self.epoch = epoch
                epoch_start_time = time.time()
                
                # Epoch start callback
                self.callback_handler.on_epoch_begin(self, epoch)
                
                # Train
                train_loss, train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validate
                val_loss, val_metrics = self.validate(val_loader)
                
                # Update history
                self._update_history(
                    train_loss, train_metrics,
                    val_loss, val_metrics,
                    time.time() - epoch_start_time
                )
                
                # Epoch end callback
                self.callback_handler.on_epoch_end(
                    self, epoch, train_metrics, val_metrics
                )
                
                # Check early stopping
                if self.callback_handler.stop_training:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Training end callback
            self.callback_handler.on_train_end(self)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.callback_handler.on_train_end(self)
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.callback_handler.on_exception(self, e)
            raise
        
        return dict(self.history)
    
    def _update_history(
        self,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: float,
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['epoch_time'].append(epoch_time)
        
        # Add metrics
        for key, value in train_metrics.items():
            self.history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            self.history[f'val_{key}'].append(value)
        
        # Add learning rate
        if self.optimizer:
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
    
    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        **kwargs
    ):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'history': dict(self.history),
            'config': self.config.to_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        load_optimizer: bool = True,
        strict: bool = True
    ):
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load EMA
        if self.ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.history = defaultdict(list, checkpoint.get('history', {}))
        
        logger.info(f"Resumed from epoch {self.epoch}")


@register_trainer("retfound")
class RETFoundTrainer(BaseTrainer):
    """Advanced trainer for RETFound with all optimizations"""
    
    def __init__(
        self,
        model: nn.Module,
        config: RETFoundConfig,
        device: Union[str, torch.device] = 'cuda'
    ):
        """Initialize RETFound trainer"""
        super().__init__(model, config, device)
        
        # Metrics
        self.train_metrics = OphthalmologyMetrics(
            config.num_classes,
            device=self.device
        )
        self.val_metrics = OphthalmologyMetrics(
            config.num_classes,
            device=self.device
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Compile model if available
        if config.use_compile and hasattr(torch, 'compile'):
            logger.info(f"Compiling model with mode: {config.compile_mode}")
            self.model = torch.compile(self.model, mode=config.compile_mode)
        
        # MixUp/CutMix
        self.mixup_cutmix = None
        if config.use_mixup or config.use_cutmix:
            self.mixup_cutmix = MixupCutmixTransform(
                mixup_alpha=config.mixup_alpha,
                cutmix_alpha=config.cutmix_alpha,
                mixup_prob=config.mixup_prob,
                cutmix_prob=config.cutmix_prob
            )
    
    def setup_training(self, train_dataset=None):
        """Setup training components"""
        logger.info("Setting up training components...")
        
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            self.config,
            train_dataset=train_dataset
        )
        
        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config
        )
        
        # Create loss function
        self.criterion = create_loss_function(
            self.config,
            train_dataset=train_dataset,
            device=self.device
        )
        
        # Setup EMA
        if self.config.use_ema:
            self.ema = EMA(
                self.model,
                decay=self.config.ema_decay,
                update_after_step=self.config.ema_update_after_step,
                update_every=self.config.ema_update_every
            )
        
        # Setup callbacks
        callbacks = create_callbacks(self.config, trainer=self)
        for callback in callbacks:
            self.callback_handler.add_callback(callback)
        
        logger.info("Training setup complete")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        num_batches = len(train_loader)
        accumulation_steps = self.config.gradient_accumulation
        
        # Memory monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Batch start callback
            self.callback_handler.on_batch_begin(self, batch_idx)
            
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            loss, outputs = self._forward_pass(images, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            self._backward_pass(loss)
            
            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0:
                self._optimizer_step()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * accumulation_steps
            
            # Only update metrics for non-mixed batches
            if not hasattr(self, '_mixed_batch') or not self._mixed_batch:
                self.train_metrics.update(outputs.detach(), labels)
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}',
                })
            
            # Batch end callback
            self.callback_handler.on_batch_end(
                self, batch_idx, loss.item(), outputs, labels
            )
            
            # Clear cache periodically
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        
        return avg_loss, metrics
    
    def _forward_pass(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional MixUp/CutMix"""
        self._mixed_batch = False
        
        # Apply MixUp/CutMix if enabled
        if self.training and self.mixup_cutmix:
            images, labels_a, labels_b, lam, method = self.mixup_cutmix(images, labels)
            self._mixed_batch = method != 'none'
            self._mix_params = (labels_a, labels_b, lam)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_amp, dtype=self.config.amp_dtype):
            outputs = self.model(images)
            
            # Calculate loss
            if self._mixed_batch:
                criterion = MixupCutmixCriterion(self.criterion)
                loss = criterion(outputs, *self._mix_params)
            else:
                loss = self.criterion(outputs, labels)
        
        return loss, outputs
    
    def _backward_pass(self, loss: torch.Tensor):
        """Backward pass with gradient scaling"""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Optimizer step with gradient clipping and SAM"""
        # Unscale gradients
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip
            )
        else:
            grad_norm = 0
        
        # Optimizer step
        if isinstance(self.optimizer, SAM):
            # SAM requires closure
            def closure():
                return self._forward_pass_closure()
            
            if self.scaler:
                self.scaler.step(self.optimizer, closure=closure)
                self.scaler.update()
            else:
                self.optimizer.step(closure=closure)
        else:
            # Standard optimizer
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Update EMA
        if self.ema:
            self.ema.update()
    
    def _forward_pass_closure(self):
        """Closure for SAM optimizer"""
        # Re-compute forward pass for SAM
        # This is a simplified version - in practice, you'd want to
        # store the batch data and reuse it
        return 0.0  # Placeholder
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        # Use EMA model if available
        model_to_eval = self.ema.ema_model if self.ema else self.model
        model_to_eval.eval()
        
        self.val_metrics.reset()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(enabled=self.config.use_amp, dtype=self.config.amp_dtype):
                    outputs = model_to_eval(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                self.val_metrics.update(outputs, labels)
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        
        return avg_loss, metrics
    
    def predict(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on a dataset"""
        model_to_eval = self.ema.ema_model if self.ema else self.model
        model_to_eval.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Prediction"):
                images = images.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config.use_amp, dtype=self.config.amp_dtype):
                    outputs = model_to_eval(images)
                
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_preds), torch.cat(all_labels)


def create_trainer(
    trainer_name: str,
    model: nn.Module,
    config: RETFoundConfig,
    **kwargs
) -> BaseTrainer:
    """
    Create a trainer from registry
    
    Args:
        trainer_name: Name of the trainer
        model: Model to train
        config: Configuration object
        **kwargs: Additional arguments
        
    Returns:
        Trainer instance
    """
    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Trainer '{trainer_name}' not found. "
            f"Available trainers: {list(TRAINER_REGISTRY.keys())}"
        )
    
    trainer_class = TRAINER_REGISTRY.get(trainer_name)
    return trainer_class(model, config, **kwargs)
