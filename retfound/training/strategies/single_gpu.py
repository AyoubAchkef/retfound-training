"""
Single GPU Training Strategy
===========================

Standard single GPU training implementation.
"""

import logging
from typing import Any, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler

from .base import TrainingStrategy

logger = logging.getLogger(__name__)


class SingleGPUStrategy(TrainingStrategy):
    """
    Single GPU training strategy
    
    This is the standard training approach for a single GPU.
    """
    
    def __init__(self, config: Any):
        """
        Initialize single GPU strategy
        
        Args:
            config: Training configuration
        """
        super().__init__(config)
        
        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            # Set default GPU if specified
            if hasattr(config, 'gpu_id'):
                torch.cuda.set_device(config.gpu_id)
        else:
            self._device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
        
        self._rank = 0
        self._world_size = 1
    
    def setup(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Tuple[nn.Module, DataLoader, Optional[DataLoader]]:
        """
        Setup single GPU training
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Tuple of (model, train_loader, val_loader)
        """
        # Move model to device
        model = model.to(self._device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model moved to {self._device}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset, is_train=True)
        val_loader = None
        if val_dataset is not None:
            val_loader = self._create_dataloader(val_dataset, is_train=False)
        
        self._setup_complete = True
        return model, train_loader, val_loader
    
    def _create_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
        """Create data loader with appropriate settings"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if is_train else self.config.batch_size * 2,
            shuffle=is_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True,
            drop_last=is_train,
            persistent_workers=self.config.persistent_workers if hasattr(self.config, 'persistent_workers') else True,
            prefetch_factor=self.config.prefetch_factor if hasattr(self.config, 'prefetch_factor') else 2
        )
    
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None
    ):
        """
        Perform backward pass
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            scaler: GradScaler for mixed precision
        """
        if scaler is not None:
            # Mixed precision backward
            scaler.scale(loss).backward()
        else:
            # Standard backward
            loss.backward()
    
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
        grad_clip: Optional[float] = None
    ):
        """
        Perform optimizer step
        
        Args:
            optimizer: Optimizer
            scaler: GradScaler for mixed precision
            grad_clip: Gradient clipping value
        """
        if scaler is not None:
            # Unscale gradients
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in optimizer.param_groups[0]['params'] if p.grad is not None],
                    grad_clip
                )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in optimizer.param_groups[0]['params'] if p.grad is not None],
                    grad_clip
                )
            
            # Standard optimizer step
            optimizer.step()
    
    def reduce_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Reduce metrics (no-op for single GPU)
        
        Args:
            metrics: Dictionary of metric tensors
            
        Returns:
            Dictionary of metric values
        """
        return {
            name: value.item() if torch.is_tensor(value) else value
            for name, value in metrics.items()
        }
    
    @property
    def is_main_process(self) -> bool:
        """Always True for single GPU"""
        return True
    
    @property
    def world_size(self) -> int:
        """Always 1 for single GPU"""
        return self._world_size
    
    @property
    def rank(self) -> int:
        """Always 0 for single GPU"""
        return self._rank
    
    @property
    def device(self) -> torch.device:
        """Get device"""
        return self._device
    
    def barrier(self):
        """No-op for single GPU"""
        pass
    
    def cleanup(self):
        """Cleanup GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()