"""
Base Training Strategy
=====================

Abstract base class for different training strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies
    
    This defines the interface that all training strategies must implement,
    whether single GPU, multi-GPU, distributed, etc.
    """
    
    def __init__(self, config: Any):
        """
        Initialize training strategy
        
        Args:
            config: Training configuration
        """
        self.config = config
        self._setup_complete = False
    
    @abstractmethod
    def setup(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Tuple[nn.Module, DataLoader, Optional[DataLoader]]:
        """
        Setup the training strategy
        
        This method should:
        1. Wrap the model if necessary (e.g., DDP)
        2. Create data loaders with appropriate samplers
        3. Setup any distributed communication if needed
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Tuple of (wrapped_model, train_loader, val_loader)
        """
        pass
    
    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None
    ):
        """
        Perform backward pass
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            scaler: GradScaler for mixed precision
        """
        pass
    
    @abstractmethod
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
        grad_clip: Optional[float] = None
    ):
        """
        Perform optimizer step
        
        Args:
            optimizer: Optimizer
            scaler: GradScaler for mixed precision
            grad_clip: Gradient clipping value
        """
        pass
    
    @abstractmethod
    def reduce_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Reduce metrics across all processes
        
        Args:
            metrics: Dictionary of metric tensors
            
        Returns:
            Dictionary of reduced metric values
        """
        pass
    
    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        pass
    
    @property
    @abstractmethod
    def world_size(self) -> int:
        """Get total number of processes"""
        pass
    
    @property
    @abstractmethod
    def rank(self) -> int:
        """Get rank of current process"""
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get device for current process"""
        pass
    
    def cleanup(self):
        """Cleanup any resources (e.g., distributed process groups)"""
        pass
    
    def barrier(self):
        """Synchronization barrier for distributed training"""
        pass
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filepath: str,
        is_best: bool = False
    ):
        """
        Save checkpoint (only on main process for distributed)
        
        Args:
            checkpoint: Checkpoint dictionary
            filepath: Path to save checkpoint
            is_best: Whether this is the best model
        """
        if self.is_main_process:
            torch.save(checkpoint, filepath)
            if is_best:
                import shutil
                best_path = filepath.replace('.pth', '_best.pth')
                shutil.copy2(filepath, best_path)
    
    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            filepath: Path to checkpoint
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def print(self, message: str):
        """Print only on main process"""
        if self.is_main_process:
            logger.info(message)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get strategy state dict for checkpointing"""
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load strategy state dict from checkpoint"""
        pass