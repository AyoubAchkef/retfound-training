"""Base model class for RETFound Training Framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

from retfound.core.exceptions import ModelError

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the framework.
    
    This class defines the interface that all models must implement
    and provides common functionality.
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
        
    @abstractmethod
    def build(self) -> None:
        """Build the model architecture.
        
        This method should define all layers and modules.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
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
        
        # Load weights
        msg = self.load_state_dict(state_dict, strict=strict)
        
        logger.info(f"Loaded weights: missing_keys={len(msg.missing_keys)}, "
                   f"unexpected_keys={len(msg.unexpected_keys)}")
        
        return {
            'missing_keys': msg.missing_keys,
            'unexpected_keys': msg.unexpected_keys,
            'checkpoint_info': {k: v for k, v in checkpoint.items() 
                               if k not in ['model_state_dict', 'state_dict', 'model']}
        }
    
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
    
    def get_layer_groups(self, num_groups: int = 4) -> list[list[nn.Module]]:
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
    ) -> list[dict]:
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
            f"Model: {self.__class__.__name__}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
        ]
        
        if hasattr(self.config, 'model'):
            summary.append(f"Architecture: {self.config.model.type}")
            summary.append(f"Number of classes: {self.config.model.num_classes}")
        
        return "\n".join(summary)
    
    def __repr__(self) -> str:
        """String representation of the model.
        
        Returns:
            String representation
        """
        return self.summary()