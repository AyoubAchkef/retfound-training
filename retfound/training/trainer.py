"""
Trainer Classes for RETFound - Dataset v6.1
==========================================

Implements the main training logic with support for advanced features
like mixed precision, gradient accumulation, and callback system.
Updated for dataset v6.1 with 28 classes and critical pathology monitoring.
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
import numpy as np

from ..core.config import RETFoundConfig
from ..core.registry import Registry
from ..core.constants import (
    NUM_TOTAL_CLASSES, UNIFIED_CLASS_NAMES,
    CRITICAL_CONDITIONS, CLASS_WEIGHTS_V61
)
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
        self.output_dir = self.getattr(config, 'output_path', 'outputs')
        self.checkpoint_dir = self.getattr(config, 'checkpoint_path', 'checkpoints')
        
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
            for epoch in range(start_epoch, self.config.training.epochs):
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
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
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
@register_trainer("retfound_v61")
class RETFoundTrainer(BaseTrainer):
    """Advanced trainer for RETFound with dataset v6.1 support"""
    
    def __init__(
        self,
        model: nn.Module,
        config: RETFoundConfig,
        device: Union[str, torch.device] = 'cuda'
    ):
        """Initialize RETFound trainer"""
        super().__init__(model, config, device)
        
        # Get number of classes from model or config
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        elif hasattr(config, 'model') and hasattr(config.model, 'num_classes'):
            num_classes = config.model.num_classes
        else:
            num_classes = NUM_TOTAL_CLASSES  # Default to 28 for v6.1
        
        # Metrics
        self.train_metrics = OphthalmologyMetrics(
            num_classes=num_classes,
            dataset_version="v6.1" if num_classes == 28 else "v4.0",
            modality=getattr(config, 'modality', None) if hasattr(config, 'data') and hasattr(config.data, 'modality') else None,
            monitor_critical=True
        )
        self.val_metrics = OphthalmologyMetrics(
            num_classes=num_classes,
            dataset_version="v6.1" if num_classes == 28 else "v4.0",
            modality=getattr(config, 'modality', None) if hasattr(config, 'data') and hasattr(config.data, 'modality') else None,
            monitor_critical=True
        )
        
        # Dataset v6.1 specific monitoring
        self.critical_class_indices = self._get_critical_class_indices()
        self.minority_class_indices = self._get_minority_class_indices()
        
        # Mixed precision
        if hasattr(config, 'optimization'):
            use_amp = config.optimization.use_amp
            amp_dtype = config.optimization.amp_dtype
        else:
            use_amp = getattr(config, 'use_amp', True)
            amp_dtype = getattr(config, 'amp_dtype', None)
        
        self.scaler = GradScaler() if use_amp else None
        self.amp_dtype = self._get_amp_dtype(amp_dtype)
        
        # Compile model if available
        if hasattr(config, 'optimization') and config.optimization.use_compile:
            compile_mode = config.optimization.compile_mode
        else:
            compile_mode = getattr(config, 'compile_mode', 'default')
        
        if hasattr(torch, 'compile') and getattr(config, 'use_compile', False):
            logger.info(f"Compiling model with mode: {compile_mode}")
            self.model = torch.compile(self.model, mode=compile_mode)
        
        # MixUp/CutMix
        self.mixup_cutmix = None
        if hasattr(config, 'augmentation'):
            use_mixup = config.augmentation.use_mixup
            use_cutmix = config.augmentation.use_cutmix
            mixup_config = {
                'mixup_alpha': config.augmentation.mixup_alpha,
                'cutmix_alpha': config.augmentation.cutmix_alpha,
                'mixup_prob': config.augmentation.mixup_prob,
                'cutmix_prob': config.augmentation.cutmix_prob
            }
        else:
            use_mixup = getattr(config, 'use_mixup', True)
            use_cutmix = getattr(config, 'use_cutmix', True)
            mixup_config = {
                'mixup_alpha': getattr(config, 'mixup_alpha', 0.8),
                'cutmix_alpha': getattr(config, 'cutmix_alpha', 1.0),
                'mixup_prob': getattr(config, 'mixup_prob', 0.5),
                'cutmix_prob': getattr(config, 'cutmix_prob', 0.5)
            }
        
        if use_mixup or use_cutmix:
            self.mixup_cutmix = MixupCutmixTransform(**mixup_config)
    
    def _get_amp_dtype(self, amp_dtype: Optional[str]) -> Optional[torch.dtype]:
        """Get AMP dtype from string"""
        if amp_dtype == 'float16':
            return torch.float16
        elif amp_dtype == 'bfloat16':
            return torch.bfloat16
        return None
    
    def _get_critical_class_indices(self) -> Dict[str, List[int]]:
        """Get indices of critical classes for monitoring"""
        critical_indices = {}
        
        for condition, info in CRITICAL_CONDITIONS.items():
            if 'unified_idx' in info:
                critical_indices[condition] = info['unified_idx']
        
        return critical_indices
    
    def _get_minority_class_indices(self) -> Dict[str, int]:
        """Get indices of minority classes from v6.1"""
        minority_indices = {}
        
        # Mapping dynamique basé sur les noms de classes
        for class_name, weight in CLASS_WEIGHTS_V61.items():
            # Recherche dans les noms de classes unifiées
            for idx, unified_name in enumerate(UNIFIED_CLASS_NAMES):
                if class_name.replace('_', ' ').lower() in unified_name.lower():
                    minority_indices[class_name] = idx
                    break
            
            # Fallback pour les cas spéciaux
            if class_name not in minority_indices:
                if "ERM" in class_name:
                    minority_indices[class_name] = 22  # OCT_ERM
                elif "RVO_OCT" in class_name:
                    minority_indices[class_name] = 25  # OCT_RVO
                elif "RAO_OCT" in class_name:
                    minority_indices[class_name] = 27  # OCT_RAO
                elif "Myopia_Degenerative" in class_name:
                    minority_indices[class_name] = 12  # Fundus_Myopia_Degenerative
        
        return minority_indices
    
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
        
        # Create loss function with class weights for v6.1
        self.criterion = create_loss_function(
            self.config,
            train_dataset=train_dataset,
            device=self.device
        )
        
        # Setup EMA
        if hasattr(self.config, 'optimization') and self.config.optimization.use_ema:
            ema_config = {
                'decay': self.config.optimization.ema_decay,
                'update_after_step': self.config.optimization.ema_update_after_step,
                'update_every': self.config.optimization.ema_update_every
            }
        else:
            ema_config = {
                'decay': getattr(self.config, 'ema_decay', 0.9999),
                'update_after_step': getattr(self.config, 'ema_update_after_step', 100),
                'update_every': getattr(self.config, 'ema_update_every', 10)
            }
        
        if getattr(self.config, 'use_ema', True):
            self.ema = EMA(self.model, **ema_config)
        
        # Setup callbacks
        callbacks = create_callbacks(self.config, trainer=self)
        for callback in callbacks:
            self.callback_handler.add_callback(callback)
        
        logger.info("Training setup complete")
        logger.info(f"Number of classes: {self.train_metrics.num_classes}")
        logger.info(f"Critical conditions monitored: {list(self.critical_class_indices.keys())}")
        logger.info(f"Minority classes monitored: {list(self.minority_class_indices.keys())}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with v6.1 monitoring"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        # Get gradient accumulation steps
        if hasattr(self.config, 'training'):
            accumulation_steps = self.config.training.gradient_accumulation_steps
        else:
            accumulation_steps = getattr(self.config, 'gradient_accumulation', 1)
        
        # Memory monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.training.epochs if hasattr(self.config, 'training') else self.getattr(config, 'epochs', 50)}")
        
        # Track per-class performance
        class_correct = torch.zeros(self.train_metrics.num_classes)
        class_total = torch.zeros(self.train_metrics.num_classes)
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 2:
                images, labels = batch
                metadata = None
            else:
                images, labels, metadata = batch
            
            # Batch start callback
            self.callback_handler.on_batch_begin(self, batch_idx)
            
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            loss, outputs = self._forward_pass(images, labels, metadata)
            
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
                
                # Track per-class accuracy
                _, preds = outputs.max(1)
                correct = preds.eq(labels).float()
                for i in range(labels.size(0)):
                    class_correct[labels[i]] += correct[i]
                    class_total[labels[i]] += 1
            
            # Update progress bar
            if batch_idx % self.getattr(config, 'log_interval', 10) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                gpu_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu': f'{gpu_memory:.1f}GB'
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
        
        # Add per-class accuracy for critical conditions
        self._log_critical_class_performance(class_correct, class_total, prefix='train')
        
        return avg_loss, metrics
    
    def _forward_pass(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional MixUp/CutMix and modality support"""
        self._mixed_batch = False
        
        # Apply MixUp/CutMix if enabled
        if self.training and self.mixup_cutmix:
            images, labels_a, labels_b, lam, method = self.mixup_cutmix(images, labels)
            self._mixed_batch = method != 'none'
            self._mix_params = (labels_a, labels_b, lam)
        
        # Gestion plus robuste des métadonnées
        modality = None
        if metadata:
            if isinstance(metadata, dict) and 'modality' in metadata:
                modality = metadata['modality']
            elif isinstance(metadata, list) and len(metadata) > 0:
                if isinstance(metadata[0], dict) and 'modality' in metadata[0]:
                    modality = metadata[0]['modality']
        
        # Forward pass with mixed precision
        with autocast(enabled=self.scaler is not None, dtype=self.amp_dtype):
            # Pass modality hint if model supports it
            if hasattr(self.model, 'forward') and 'modality' in self.model.forward.__code__.co_varnames:
                outputs = self.model(images, modality=modality)
            else:
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
        gradient_clip = self.config.optimization.gradient_clip if hasattr(self.config, 'optimization') else self.getattr(config, 'gradient_clip', 1.0)
        
        if gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=gradient_clip
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
        """Validate model with v6.1 critical class monitoring"""
        # Use EMA model if available
        model_to_eval = self.ema.ema_model if self.ema else self.model
        model_to_eval.eval()
        
        self.val_metrics.reset()
        total_loss = 0
        num_batches = len(val_loader)
        
        # Track per-class performance
        class_correct = torch.zeros(self.val_metrics.num_classes)
        class_total = torch.zeros(self.val_metrics.num_classes)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle different batch formats
                if len(batch) == 2:
                    images, labels = batch
                    metadata = None
                else:
                    images, labels, metadata = batch
                
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Get modality hint if available
                modality = None
                if metadata and 'modality' in metadata:
                    modality = metadata['modality']
                
                # Forward pass
                with autocast(enabled=self.scaler is not None, dtype=self.amp_dtype):
                    if hasattr(model_to_eval, 'forward') and 'modality' in model_to_eval.forward.__code__.co_varnames:
                        outputs = model_to_eval(images, modality=modality)
                    else:
                        outputs = model_to_eval(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                self.val_metrics.update(outputs, labels)
                
                # Track per-class accuracy
                _, preds = outputs.max(1)
                correct = preds.eq(labels).float()
                for i in range(labels.size(0)):
                    class_correct[labels[i]] += correct[i]
                    class_total[labels[i]] += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        
        # Add critical class performance
        critical_metrics = self._log_critical_class_performance(
            class_correct, class_total, prefix='val'
        )
        metrics.update(critical_metrics)
        
        return avg_loss, metrics
    
    def _log_critical_class_performance(
        self, 
        class_correct: torch.Tensor, 
        class_total: torch.Tensor,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """Log performance for critical and minority classes"""
        metrics = {}
        
        # Log critical conditions
        logger.info(f"\n{prefix.upper()} - Critical Conditions Performance:")
        for condition, indices in self.critical_class_indices.items():
            condition_correct = sum(class_correct[idx] for idx in indices if idx < len(class_correct))
            condition_total = sum(class_total[idx] for idx in indices if idx < len(class_total))
            
            if condition_total > 0:
                accuracy = (condition_correct / condition_total).item()
                sensitivity_target = CRITICAL_CONDITIONS[condition]['min_sensitivity']
                
                logger.info(
                    f"  {condition}: {accuracy:.3f} "
                    f"(target: {sensitivity_target:.3f}) "
                    f"[{int(condition_correct)}/{int(condition_total)}]"
                )
                
                metrics[f'{prefix}_{condition}_acc'] = accuracy
                
                # Alert if below target
                if accuracy < sensitivity_target:
                    logger.warning(
                        f"  ⚠️ {condition} below target sensitivity! "
                        f"{accuracy:.3f} < {sensitivity_target:.3f}"
                    )
        
        # Log minority classes
        logger.info(f"\n{prefix.upper()} - Minority Classes Performance:")
        for class_name, idx in self.minority_class_indices.items():
            if idx < len(class_correct) and class_total[idx] > 0:
                accuracy = (class_correct[idx] / class_total[idx]).item()
                logger.info(
                    f"  {class_name}: {accuracy:.3f} "
                    f"[{int(class_correct[idx])}/{int(class_total[idx])}]"
                )
                metrics[f'{prefix}_{class_name}_acc'] = accuracy
        
        return metrics
    
    def predict(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on a dataset"""
        model_to_eval = self.ema.ema_model if self.ema else self.model
        model_to_eval.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction"):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels, _ = batch
                
                images = images.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.scaler is not None, dtype=self.amp_dtype):
                    outputs = model_to_eval(images)
                
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_preds), torch.cat(all_labels)
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def log_summary(self):
        """Log training summary with focus on v6.1 metrics"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY - Dataset v6.1")
        logger.info("="*60)
        
        # Best metrics
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best metric: {self.best_metric:.4f}")
        
        # Final performance on critical conditions
        if 'val_RAO_acc' in self.history:
            logger.info("\nFinal Critical Conditions Performance:")
            for condition in self.critical_class_indices.keys():
                key = f'val_{condition}_acc'
                if key in self.history and self.history[key]:
                    final_acc = self.history[key][-1]
                    logger.info(f"  {condition}: {final_acc:.3f}")
        
        logger.info("="*60)


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
