"""
Distributed Data Parallel (DDP) Training Strategy
================================================

Multi-GPU training using PyTorch DDP.
"""

import os
import logging
from typing import Any, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler

from .base import TrainingStrategy

logger = logging.getLogger(__name__)


class DDPStrategy(TrainingStrategy):
    """
    Distributed Data Parallel training strategy
    
    Supports multi-GPU training on single or multiple nodes.
    """
    
    def __init__(self, config: Any):
        """
        Initialize DDP strategy
        
        Args:
            config: Training configuration
        """
        super().__init__(config)
        
        # Get distributed settings from environment or config
        self._rank = int(os.environ.get('RANK', 0))
        self._local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self._world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Override with config if provided
        if hasattr(config, 'rank'):
            self._rank = config.rank
        if hasattr(config, 'local_rank'):
            self._local_rank = config.local_rank
        if hasattr(config, 'world_size'):
            self._world_size = config.world_size
        
        # Setup device
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            self._device = torch.device(f'cuda:{self._local_rank}')
        else:
            raise RuntimeError("DDP requires CUDA")
        
        # Initialize process group
        self._init_process_group()
    
    def _init_process_group(self):
        """Initialize distributed process group"""
        if not dist.is_initialized():
            # Get backend
            backend = self.config.dist_backend if hasattr(self.config, 'dist_backend') else 'nccl'
            
            # Get init method
            init_method = os.environ.get('INIT_METHOD', 'env://')
            
            # Initialize
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self._world_size,
                rank=self._rank
            )
            
            logger.info(
                f"Initialized process group: "
                f"rank={self._rank}, local_rank={self._local_rank}, "
                f"world_size={self._world_size}"
            )
    
    def setup(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Tuple[nn.Module, DataLoader, Optional[DataLoader]]:
        """
        Setup DDP training
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Tuple of (ddp_model, train_loader, val_loader)
        """
        # Move model to device
        model = model.to(self._device)
        
        # Wrap with DDP
        find_unused = self.config.find_unused_parameters if hasattr(self.config, 'find_unused_parameters') else False
        
        ddp_model = DDP(
            model,
            device_ids=[self._local_rank],
            output_device=self._local_rank,
            find_unused_parameters=find_unused
        )
        
        # Log model info (only on main process)
        if self.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model wrapped with DDP")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create data loaders with distributed samplers
        train_loader = self._create_dataloader(train_dataset, is_train=True)
        val_loader = None
        if val_dataset is not None:
            val_loader = self._create_dataloader(val_dataset, is_train=False)
        
        self._setup_complete = True
        return ddp_model, train_loader, val_loader
    
    def _create_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
        """Create data loader with distributed sampler"""
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=is_train,
            drop_last=is_train
        )
        
        # Create data loader
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if is_train else self.config.batch_size * 2,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True,
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
        Reduce metrics across all processes
        
        Args:
            metrics: Dictionary of metric tensors
            
        Returns:
            Dictionary of reduced metric values
        """
        reduced_metrics = {}
        
        for name, value in metrics.items():
            if torch.is_tensor(value):
                # Move to device if needed
                if value.device != self._device:
                    value = value.to(self._device)
                
                # All-reduce
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value = value / self._world_size
                
                reduced_metrics[name] = value.item()
            else:
                reduced_metrics[name] = value
        
        return reduced_metrics
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self._rank == 0
    
    @property
    def world_size(self) -> int:
        """Get total number of processes"""
        return self._world_size
    
    @property
    def rank(self) -> int:
        """Get rank of current process"""
        return self._rank
    
    @property
    def device(self) -> torch.device:
        """Get device for current process"""
        return self._device
    
    def barrier(self):
        """Synchronization barrier"""
        if dist.is_initialized():
            dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed process group"""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filepath: str,
        is_best: bool = False
    ):
        """Save checkpoint only on main process"""
        if self.is_main_process:
            # For DDP, extract the module from the wrapper
            if 'model_state_dict' in checkpoint:
                # If the model is wrapped in DDP, get the underlying module
                model_state = checkpoint['model_state_dict']
                if hasattr(model_state, 'module'):
                    checkpoint['model_state_dict'] = model_state.module.state_dict()
            
            super().save_checkpoint(checkpoint, filepath, is_best)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get strategy state dict"""
        return {
            'rank': self._rank,
            'world_size': self._world_size,
            'local_rank': self._local_rank
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load strategy state dict"""
        # Verify configuration matches
        if state_dict.get('world_size') != self._world_size:
            logger.warning(
                f"World size mismatch: checkpoint has {state_dict.get('world_size')}, "
                f"current has {self._world_size}"
            )