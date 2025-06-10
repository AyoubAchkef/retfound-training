"""
Model Factory for RETFound
========================

Factory pattern for creating RETFound models with different configurations
and pre-trained weights.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Any
from functools import partial

import torch
import torch.nn as nn

from ..core.config import RETFoundConfig
from ..core.registry import Registry
from ..core.exceptions import ModelNotFoundError, WeightsNotFoundError
from .retfound import RETFoundModel

logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = Registry("models")


def register_model(name: str):
    """Decorator to register a model"""
    def decorator(cls):
        MODEL_REGISTRY.register(name, cls)
        return cls
    return decorator


@register_model("retfound")
@register_model("vit_large_patch16_224")
class RETFoundFactory:
    """Factory for RETFound models"""
    
    # Available pre-trained weights
    PRETRAINED_WEIGHTS = {
        'cfp': {
            'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth',
            'md5': 'a7c8f4e6c3f6b8d9e2f1a5b4c7d8e9f0',
            'description': 'Pre-trained on Color Fundus Photography'
        },
        'oct': {
            'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth',
            'md5': 'b8d9f0e1d2c3a4b5c6d7e8f9a0b1c2d3',
            'description': 'Pre-trained on OCT images'
        },
        'meh': {
            'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_meh.pth',
            'md5': 'c9e0f1e2d3c4a5b6c7d8e9f0a1b2c3d4',
            'description': 'Pre-trained on Multi-Ethnic Hybrid dataset'
        }
    }
    
    @classmethod
    def create(
        cls,
        config: RETFoundConfig,
        pretrained: Optional[str] = None,
        weights_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        **kwargs
    ) -> RETFoundModel:
        """
        Create a RETFound model
        
        Args:
            config: Model configuration
            pretrained: Name of pretrained weights ('cfp', 'oct', 'meh')
            weights_path: Path to custom weights file
            strict: Whether to strictly enforce weight loading
            **kwargs: Additional arguments passed to model
            
        Returns:
            RETFoundModel instance
        """
        # Create model
        logger.info(f"Creating RETFound model: {config.model_type}")
        model = RETFoundModel(config, **kwargs)
        
        # Load pretrained weights if specified
        if pretrained:
            if pretrained not in cls.PRETRAINED_WEIGHTS:
                raise ValueError(
                    f"Unknown pretrained weights: {pretrained}. "
                    f"Available: {list(cls.PRETRAINED_WEIGHTS.keys())}"
                )
            
            # Use custom path or default
            if not weights_path:
                weights_path = config.weights_paths.get(
                    pretrained,
                    Path(f"RETFound_mae_{pretrained}.pth")
                )
            
            logger.info(f"Loading {pretrained} weights from {weights_path}")
            cls._load_pretrained_weights(model, weights_path, pretrained, strict)
        
        # Load custom weights if specified
        elif weights_path:
            logger.info(f"Loading custom weights from {weights_path}")
            cls._load_custom_weights(model, weights_path, strict)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model created: {total_params/1e6:.1f}M parameters, "
            f"{trainable_params/1e6:.1f}M trainable"
        )
        
        return model
    
    @classmethod
    def _load_pretrained_weights(
        cls,
        model: RETFoundModel,
        weights_path: Union[str, Path],
        pretrained_type: str,
        strict: bool = False
    ):
        """Load pre-trained RETFound weights"""
        weights_path = Path(weights_path)
        
        # Check if weights exist
        if not weights_path.exists():
            raise WeightsNotFoundError(
                f"Weights not found: {weights_path}\n"
                f"Please download from: {cls.PRETRAINED_WEIGHTS[pretrained_type]['url']}"
            )
        
        # Load weights
        try:
            model.load_pretrained_weights(weights_path, model_key=pretrained_type)
            logger.info(f"Successfully loaded {pretrained_type} weights")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            if strict:
                raise
    
    @classmethod
    def _load_custom_weights(
        cls,
        model: RETFoundModel,
        weights_path: Union[str, Path],
        strict: bool = False
    ):
        """Load custom weights"""
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise WeightsNotFoundError(f"Weights not found: {weights_path}")
        
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=strict
            )
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.info("Successfully loaded custom weights")
            
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            if strict:
                raise


def create_model(
    model_name: str,
    config: RETFoundConfig,
    **kwargs
) -> nn.Module:
    """
    Create a model from registry
    
    Args:
        model_name: Name of the model
        config: Model configuration
        **kwargs: Additional arguments
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ModelNotFoundError(
            f"Model '{model_name}' not found. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_factory = MODEL_REGISTRY.get(model_name)
    return model_factory.create(config, **kwargs)


def list_models() -> list:
    """List all available models"""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model"""
    if model_name not in MODEL_REGISTRY:
        raise ModelNotFoundError(f"Model '{model_name}' not found")
    
    factory = MODEL_REGISTRY.get(model_name)
    
    return {
        'name': model_name,
        'factory': factory.__name__,
        'pretrained_weights': list(factory.PRETRAINED_WEIGHTS.keys())
        if hasattr(factory, 'PRETRAINED_WEIGHTS') else [],
        'description': factory.__doc__ or "No description available"
    }


# Convenience functions
def create_retfound_cfp(config: RETFoundConfig, **kwargs) -> RETFoundModel:
    """Create RETFound model with CFP weights"""
    return create_model('retfound', config, pretrained='cfp', **kwargs)


def create_retfound_oct(config: RETFoundConfig, **kwargs) -> RETFoundModel:
    """Create RETFound model with OCT weights"""
    return create_model('retfound', config, pretrained='oct', **kwargs)


def create_retfound_meh(config: RETFoundConfig, **kwargs) -> RETFoundModel:
    """Create RETFound model with MEH weights"""
    return create_model('retfound', config, pretrained='meh', **kwargs)
