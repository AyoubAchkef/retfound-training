"""
Logging Callbacks
================

Implements various logging callbacks for monitoring training progress.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from collections import defaultdict

import torch
import numpy as np

from .base import Callback

logger = logging.getLogger(__name__)

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available")


class LoggingCallback(Callback):
    """
    Basic logging callback for console output
    """
    
    def __init__(
        self,
        log_frequency: int = 10,
        log_memory: bool = True,
        log_gradients: bool = True
    ):
        """
        Initialize logging callback
        
        Args:
            log_frequency: Log every N batches
            log_memory: Whether to log GPU memory usage
            log_gradients: Whether to log gradient norms
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.log_memory = log_memory
        self.log_gradients = log_gradients
        
        # Track batch times
        self.batch_times = []
        self.epoch_start_time = None
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Record epoch start time"""
        self.epoch_start_time = datetime.now()
        self.batch_times = []
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{trainer.config.epochs} started")
        logger.info(f"{'='*70}")
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float,
        outputs: Any,
        targets: Any
    ):
        """Log batch statistics"""
        if batch_idx % self.log_frequency == 0:
            # Calculate statistics
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            log_msg = f"Batch {batch_idx}: loss={loss:.4f}, lr={current_lr:.2e}"
            
            # Add gradient norm if available
            if self.log_gradients and hasattr(trainer, '_last_grad_norm'):
                log_msg += f", grad_norm={trainer._last_grad_norm:.3f}"
            
            # Add memory usage
            if self.log_memory and torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                log_msg += f", mem={memory_mb:.0f}/{max_memory_mb:.0f}MB"
            
            logger.info(log_msg)
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch summary"""
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"{'='*50}")
        
        # Training metrics
        logger.info("Training Metrics:")
        for key, value in sorted(train_metrics.items()):
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
        
        # Validation metrics
        logger.info("\nValidation Metrics:")
        for key, value in sorted(val_metrics.items()):
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
        
        # Time statistics
        logger.info(f"\nTime: {epoch_time:.1f}s")
        logger.info(f"{'='*50}\n")
    
    def on_train_end(self, trainer):
        """Log training summary"""
        total_time = sum(trainer.history.get('epoch_time', []))
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total epochs: {trainer.epoch + 1}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best epoch: {trainer.best_epoch}")
        logger.info(f"Best metric: {trainer.best_metric:.4f}")


class TensorBoardCallback(Callback):
    """
    TensorBoard logging callback
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = '',
        log_graph: bool = True,
        log_images: bool = True,
        log_histograms: bool = True,
        log_frequency: int = 100
    ):
        """
        Initialize TensorBoard callback
        
        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to add to run name
            log_graph: Whether to log model graph
            log_images: Whether to log images
            log_histograms: Whether to log weight histograms
            log_frequency: Frequency for detailed logging
        """
        super().__init__()
        
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available")
        
        self.log_dir = Path(log_dir)
        self.comment = comment
        self.log_graph = log_graph
        self.log_images = log_images
        self.log_histograms = log_histograms
        self.log_frequency = log_frequency
        
        self.writer = None
        self.global_step = 0
    
    def on_train_begin(self, trainer):
        """Initialize TensorBoard writer"""
        # Create unique run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{timestamp}_{self.comment}" if self.comment else timestamp
        run_dir = self.log_dir / run_name
        
        self.writer = SummaryWriter(log_dir=run_dir)
        
        # Log model graph if requested
        if self.log_graph:
            try:
                dummy_input = torch.randn(
                    1, 3,
                    trainer.config.input_size,
                    trainer.config.input_size
                ).to(trainer.device)
                self.writer.add_graph(trainer.model, dummy_input)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
        
        # Log configuration
        config_dict = trainer.config.to_dict() if hasattr(trainer.config, 'to_dict') else {}
        self.writer.add_text('config', json.dumps(config_dict, indent=2), 0)
        
        logger.info(f"TensorBoard logging to: {run_dir}")
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float,
        outputs: Any,
        targets: Any
    ):
        """Log batch metrics"""
        self.global_step += 1
        
        # Log loss
        self.writer.add_scalar('train/batch_loss', loss, self.global_step)
        
        # Log learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
        
        # Log detailed metrics periodically
        if self.global_step % self.log_frequency == 0:
            # Log gradient norms
            if hasattr(trainer, '_last_grad_norm'):
                self.writer.add_scalar(
                    'train/gradient_norm',
                    trainer._last_grad_norm,
                    self.global_step
                )
            
            # Log weight histograms
            if self.log_histograms:
                for name, param in trainer.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(
                            f'weights/{name}',
                            param.data,
                            self.global_step
                        )
                        self.writer.add_histogram(
                            f'gradients/{name}',
                            param.grad,
                            self.global_step
                        )
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch metrics"""
        # Log training metrics
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{key}', value, epoch)
        
        # Log validation metrics
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log confusion matrix if available
        if hasattr(trainer.val_metrics, 'confusion_matrix'):
            cm = trainer.val_metrics.confusion_matrix()
            if cm is not None:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(cm, cmap='Blues')
                plt.colorbar()
                plt.title('Confusion Matrix')
                self.writer.add_figure('val/confusion_matrix', fig, epoch)
                plt.close(fig)
    
    def on_train_end(self, trainer):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class WandBCallback(Callback):
    """
    Weights & Biases logging callback
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Any] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        log_model: bool = True,
        log_frequency: int = 100,
        log_images: bool = True
    ):
        """
        Initialize W&B callback
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            name: Run name
            config: Configuration to log
            tags: Tags for the run
            notes: Notes for the run
            log_model: Whether to log model artifacts
            log_frequency: Frequency for detailed logging
            log_images: Whether to log images
        """
        super().__init__()
        
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases not available")
        
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.tags = tags or []
        self.notes = notes
        self.log_model = log_model
        self.log_frequency = log_frequency
        self.log_images = log_images
        
        self.run = None
        self.global_step = 0
    
    def on_train_begin(self, trainer):
        """Initialize W&B run"""
        # Prepare config
        if self.config is None:
            self.config = trainer.config
        
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
        
        # Initialize run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name or f"retfound_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            tags=self.tags + ['retfound'],
            notes=self.notes
        )
        
        # Watch model
        wandb.watch(trainer.model, log='all', log_freq=self.log_frequency)
        
        logger.info(f"W&B run initialized: {self.run.url}")
    
    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: float,
        outputs: Any,
        targets: Any
    ):
        """Log batch metrics"""
        self.global_step += 1
        
        # Log basic metrics
        log_dict = {
            'train/batch_loss': loss,
            'train/learning_rate': trainer.optimizer.param_groups[0]['lr'],
            'global_step': self.global_step
        }
        
        # Add gradient norm if available
        if hasattr(trainer, '_last_grad_norm'):
            log_dict['train/gradient_norm'] = trainer._last_grad_norm
        
        # Add GPU metrics
        if torch.cuda.is_available():
            log_dict['system/gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            log_dict['system/gpu_utilization'] = torch.cuda.utilization()
        
        wandb.log(log_dict)
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch metrics"""
        log_dict = {'epoch': epoch}
        
        # Add all metrics
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f'train/{key}'] = value
        
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f'val/{key}'] = value
        
        # Log confusion matrix if available
        if hasattr(trainer.val_metrics, 'confusion_matrix'):
            cm = trainer.val_metrics.confusion_matrix()
            if cm is not None:
                log_dict['val/confusion_matrix'] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=trainer.val_metrics.targets,
                    preds=trainer.val_metrics.predictions,
                    class_names=trainer.val_metrics.class_names
                )
        
        wandb.log(log_dict)
    
    def on_checkpoint_save(self, trainer, checkpoint, filepath):
        """Log model checkpoint"""
        if self.log_model:
            # Save model artifact
            artifact = wandb.Artifact(
                name=f"model-{trainer.epoch}",
                type="model",
                description=f"Model checkpoint at epoch {trainer.epoch}"
            )
            artifact.add_file(filepath)
            
            # Add metadata
            artifact.metadata = {
                'epoch': trainer.epoch,
                'best_metric': trainer.best_metric,
                'metrics': checkpoint.get('metrics', {})
            }
            
            self.run.log_artifact(artifact)
    
    def on_train_end(self, trainer):
        """Finish W&B run"""
        if self.run:
            # Log final summary
            summary = {
                'best_epoch': trainer.best_epoch,
                'best_metric': trainer.best_metric,
                'total_epochs': trainer.epoch + 1,
                'total_time_hours': sum(trainer.history.get('epoch_time', [])) / 3600
            }
            
            for key, value in summary.items():
                wandb.summary[key] = value
            
            # Finish run
            self.run.finish()
