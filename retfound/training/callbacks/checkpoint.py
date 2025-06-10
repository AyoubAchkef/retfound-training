"""
Checkpoint Callback
==================

Handles model checkpointing during training.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch

from .base import Callback

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints
    
    Features:
    - Save at regular intervals
    - Save best model based on metric
    - Save last model
    - Keep only N best checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_frequency: int = 10,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        n_best: int = 3,
        filename_prefix: str = 'retfound',
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save every N epochs
            save_best: Whether to save best model
            save_last: Whether to save last model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for metric
            n_best: Number of best checkpoints to keep
            filename_prefix: Prefix for checkpoint files
            verbose: Whether to print save messages
        """
        super().__init__()
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode
        self.n_best = n_best
        self.filename_prefix = filename_prefix
        self.verbose = verbose
        
        # Track best metric
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        
        # Track saved checkpoints
        self.saved_checkpoints = []
        self.best_checkpoints = []  # (metric, epoch, filepath)
    
    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save checkpoint at end of epoch"""
        # Regular checkpoint
        if (epoch + 1) % self.save_frequency == 0:
            filepath = self.checkpoint_dir / f'{self.filename_prefix}_epoch_{epoch+1}.pth'
            self._save_checkpoint(trainer, filepath, epoch, val_metrics)
            self.saved_checkpoints.append(filepath)
        
        # Best model checkpoint
        if self.save_best and self.monitor in val_metrics:
            metric_value = val_metrics[self.monitor]
            
            is_best = False
            if self.mode == 'max' and metric_value > self.best_metric:
                is_best = True
            elif self.mode == 'min' and metric_value < self.best_metric:
                is_best = True
            
            if is_best:
                self.best_metric = metric_value
                self.best_epoch = epoch
                
                filepath = self.checkpoint_dir / f'{self.filename_prefix}_best.pth'
                self._save_checkpoint(trainer, filepath, epoch, val_metrics, is_best=True)
                
                # Track best checkpoints
                self.best_checkpoints.append((metric_value, epoch, filepath))
                self._cleanup_best_checkpoints()
                
                if self.verbose:
                    logger.info(
                        f"New best model! {self.monitor}: {metric_value:.4f} "
                        f"(previous: {self.best_metric:.4f})"
                    )
        
        # Last model checkpoint
        if self.save_last:
            filepath = self.checkpoint_dir / f'{self.filename_prefix}_last.pth'
            self._save_checkpoint(trainer, filepath, epoch, val_metrics)
    
    def _save_checkpoint(
        self,
        trainer,
        filepath: Path,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save checkpoint to file"""
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer else None,
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'scaler_state_dict': trainer.scaler.state_dict() if trainer.scaler else None,
            'metrics': metrics,
            'history': dict(trainer.history),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': trainer.config.to_dict() if hasattr(trainer.config, 'to_dict') else {},
            'is_best': is_best
        }
        
        # Add EMA if available
        if hasattr(trainer, 'ema') and trainer.ema:
            checkpoint['ema_state_dict'] = trainer.ema.state_dict()
        
        # Allow trainer to add custom data
        trainer.callback_handler.on_checkpoint_save(trainer, checkpoint, str(filepath))
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            logger.info(f"Checkpoint saved: {filepath}")
    
    def _cleanup_best_checkpoints(self):
        """Keep only N best checkpoints"""
        if len(self.best_checkpoints) <= self.n_best:
            return
        
        # Sort by metric (ascending for min, descending for max)
        reverse = (self.mode == 'max')
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Remove worst checkpoints
        while len(self.best_checkpoints) > self.n_best:
            metric, epoch, filepath = self.best_checkpoints.pop()
            if filepath.exists():
                filepath.unlink()
                if self.verbose:
                    logger.info(f"Removed checkpoint: {filepath}")
    
    def on_train_end(self, trainer):
        """Final checkpoint save"""
        # Save final model
        filepath = self.checkpoint_dir / f'{self.filename_prefix}_final.pth'
        self._save_checkpoint(
            trainer,
            filepath,
            trainer.epoch,
            trainer.val_metrics.compute() if hasattr(trainer, 'val_metrics') else {}
        )
        
        # Log summary
        logger.info(f"\nCheckpoint Summary:")
        logger.info(f"  Total checkpoints saved: {len(self.saved_checkpoints)}")
        logger.info(f"  Best epoch: {self.best_epoch}")
        logger.info(f"  Best {self.monitor}: {self.best_metric:.4f}")
        logger.info(f"  Checkpoint directory: {self.checkpoint_dir}")
    
    def state_dict(self) -> Dict[str, Any]:
        """Get callback state"""
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'saved_checkpoints': [str(p) for p in self.saved_checkpoints],
            'best_checkpoints': [
                (m, e, str(p)) for m, e, p in self.best_checkpoints
            ]
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load callback state"""
        self.best_metric = state_dict.get('best_metric', self.best_metric)
        self.best_epoch = state_dict.get('best_epoch', self.best_epoch)
        self.saved_checkpoints = [
            Path(p) for p in state_dict.get('saved_checkpoints', [])
        ]
        self.best_checkpoints = [
            (m, e, Path(p)) for m, e, p in state_dict.get('best_checkpoints', [])
        ]
