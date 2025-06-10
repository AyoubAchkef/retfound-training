"""Base model class for RETFound Training Framework - Enhanced for v6.1."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime

from retfound.core.exceptions import ModelError

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the framework.
    
    This class defines the interface that all models must implement
    and provides common functionality, with enhanced support for v6.1.
    """
    
    def __init__(self, config: Any):
        """Initialize base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self._is_built = False
        self._device = None
        self._metadata = {}  # For storing model metadata
        
        # V6.1 specific attributes
        self._dataset_version = None
        self._modality = None
        self._unified_classes = False
        
    @abstractmethod
    def build(self) -> None:
        """Build the model architecture.
        
        This method should define all layers and modules.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments (e.g., modality hint for v6.1)
            
        Returns:
            Output tensor or dictionary of outputs
        """
        pass
    
    def initialize_weights(self) -> None:
        """Initialize model weights.
        
        Default implementation uses standard initialization.
        Can be overridden by subclasses for custom initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def load_pretrained_weights(
        self,
        checkpoint_path: Path,
        strict: bool = False,
        map_location: str = 'cpu'
    ) -> Dict[str, Any]:
        """Load pretrained weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce that the keys match
            map_location: Device to map the checkpoint to
            
        Returns:
            Dictionary with loading information
        """
        if not checkpoint_path.exists():
            raise ModelError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Extract metadata if available (v6.1 support)
        if 'metadata' in checkpoint:
            self._metadata = checkpoint['metadata']
            self._dataset_version = self._metadata.get('dataset_version')
            self._modality = self._metadata.get('modality')
            self._unified_classes = self._metadata.get('unified_classes', False)
            
            logger.info(f"Loaded checkpoint metadata:")
            logger.info(f"  - Dataset version: {self._dataset_version}")
            logger.info(f"  - Modality: {self._modality}")
            logger.info(f"  - Unified classes: {self._unified_classes}")
        
        # Load weights
        msg = self.load_state_dict(state_dict, strict=strict)
        
        logger.info(f"Loaded weights: missing_keys={len(msg.missing_keys)}, "
                   f"unexpected_keys={len(msg.unexpected_keys)}")
        
        return {
            'missing_keys': msg.missing_keys,
            'unexpected_keys': msg.unexpected_keys,
            'checkpoint_info': {k: v for k, v in checkpoint.items() 
                               if k not in ['model_state_dict', 'state_dict', 'model']},
            'metadata': self._metadata
        }
    
    def save_checkpoint(
        self,
        save_path: Path,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        epoch: Optional[int] = None,
        **kwargs
    ) -> None:
        """Save model checkpoint with metadata.
        
        Args:
            save_path: Path to save checkpoint
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            epoch: Current epoch
            **kwargs: Additional items to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'metadata': self.get_metadata(),
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        # Add any additional items
        checkpoint.update(kwargs)
        
        # Save checkpoint
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        
        logger.info(f"Saved checkpoint to {save_path}")
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning.
        
        Can be overridden by subclasses to implement custom freezing logic.
        """
        frozen_params = 0
        for name, param in self.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Frozen {frozen_params:,} / {total_params:,} parameters "
                   f"({frozen_params/total_params*100:.1f}%)")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        
        logger.info("All parameters unfrozen")
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get number of parameters.
        
        Args:
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_layer_groups(self, num_groups: int = 4) -> List[List[nn.Module]]:
        """Get layer groups for differential learning rates.
        
        Args:
            num_groups: Number of groups to create
            
        Returns:
            List of layer groups
            
        Note:
            This is a simple implementation that groups sequential layers.
            Subclasses should override for architecture-specific grouping.
        """
        all_layers = list(self.children())
        if len(all_layers) <= num_groups:
            return [[layer] for layer in all_layers]
        
        # Divide layers into groups
        layers_per_group = len(all_layers) // num_groups
        groups = []
        
        for i in range(num_groups):
            start_idx = i * layers_per_group
            if i == num_groups - 1:
                # Last group gets remaining layers
                groups.append(all_layers[start_idx:])
            else:
                end_idx = start_idx + layers_per_group
                groups.append(all_layers[start_idx:end_idx])
        
        return groups
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory.
        
        Must be implemented by subclasses that support it.
        """
        logger.warning(f"{self.__class__.__name__} does not support gradient checkpointing")
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        logger.warning(f"{self.__class__.__name__} does not support gradient checkpointing")
    
    def get_optimizer_params(
        self,
        base_lr: float,
        weight_decay: float = 0.0,
        layer_decay: Optional[float] = None
    ) -> List[Dict]:
        """Get parameter groups for optimizer with optional layer-wise decay.
        
        Args:
            base_lr: Base learning rate
            weight_decay: Weight decay value
            layer_decay: Layer-wise learning rate decay factor
            
        Returns:
            List of parameter groups for optimizer
        """
        if layer_decay is None:
            # Simple parameter groups
            decay_params = []
            no_decay_params = []
            
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            return [
                {'params': decay_params, 'lr': base_lr, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'lr': base_lr, 'weight_decay': 0.0}
            ]
        else:
            # Layer-wise decay - must be implemented by subclasses
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_optimizer_params "
                "with layer_decay support"
            )
    
    def to(self, device: torch.device) -> 'BaseModel':
        """Move model to device and store device reference.
        
        Args:
            device: Target device
            
        Returns:
            Self
        """
        self._device = device
        return super().to(device)
    
    @property
    def device(self) -> Optional[torch.device]:
        """Get the device the model is on.
        
        Returns:
            Device or None if not set
        """
        if self._device is not None:
            return self._device
        
        # Try to infer from parameters
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None
    
    # V6.1 specific methods
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata including v6.1 specific information.
        
        Returns:
            Dictionary containing model metadata
        """
        metadata = {
            'model_class': self.__class__.__name__,
            'num_parameters': self.get_num_params(),
            'trainable_parameters': self.get_num_params(trainable_only=True),
        }
        
        # Add config info if available
        if hasattr(self.config, 'model'):
            metadata.update({
                'architecture': getattr(self.config.model, 'type', 'unknown'),
                'num_classes': getattr(self.config.model, 'num_classes', 'unknown'),
            })
        
        # Add v6.1 specific metadata
        if self._dataset_version:
            metadata['dataset_version'] = self._dataset_version
        if self._modality:
            metadata['modality'] = self._modality
        if self._unified_classes:
            metadata['unified_classes'] = self._unified_classes
        
        # Merge with any custom metadata
        metadata.update(self._metadata)
        
        return metadata
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set custom metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    @property
    def is_v61_model(self) -> bool:
        """Check if this is a v6.1 model.
        
        Returns:
            True if model is configured for v6.1
        """
        return (
            self._dataset_version == 'v6.1' or
            (hasattr(self.config, 'model') and 
             getattr(self.config.model, 'num_classes', 0) == 28)
        )
    
    def get_class_names(self) -> Optional[List[str]]:
        """Get class names if available.
        
        Returns:
            List of class names or None
            
        Note:
            Should be overridden by subclasses that have class name information
        """
        return None
    
    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """Get model summary.
        
        Args:
            input_shape: Input shape for summary (excluding batch dimension)
            
        Returns:
            Model summary string
        """
        total_params = self.get_num_params()
        trainable_params = self.get_num_params(trainable_only=True)
        
        summary = [
            f"{'='*60}",
            f"Model: {self.__class__.__name__}",
            f"{'='*60}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
            f"Memory footprint: ~{total_params * 4 / 1024**2:.1f} MB (float32)",
        ]
        
        if hasattr(self.config, 'model'):
            summary.extend([
                f"{'─'*60}",
                f"Architecture: {self.config.model.type}",
                f"Number of classes: {self.config.model.num_classes}",
            ])
        
        # Add v6.1 specific information
        if self.is_v61_model:
            summary.extend([
                f"{'─'*60}",
                f"Dataset version: v6.1",
                f"Unified classes: {self._unified_classes}",
                f"Modality: {self._modality or 'not specified'}",
            ])
        
        # Add metadata if available
        if self._metadata:
            summary.extend([
                f"{'─'*60}",
                "Additional metadata:",
            ])
            for key, value in self._metadata.items():
                if key not in ['model_class', 'num_parameters', 'trainable_parameters',
                              'architecture', 'num_classes', 'dataset_version', 
                              'modality', 'unified_classes']:
                    summary.append(f"  {key}: {value}")
        
        summary.append(f"{'='*60}")
        
        return "\n".join(summary)
    
    def __repr__(self) -> str:
        """String representation of the model.
        
        Returns:
            String representation
        """
        return self.summary()