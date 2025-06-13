"""
Model Factory for RETFound - Enhanced for Dataset v6.1
======================================================

Factory pattern for creating RETFound models with different configurations
and pre-trained weights, with full support for CAASI dataset v6.1.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
from functools import partial

import torch
import torch.nn as nn

from ..core.config import RETFoundConfig
from ..core.registry import Registry
from ..core.exceptions import ModelNotFoundError, WeightsNotFoundError
from ..core.constants import NUM_TOTAL_CLASSES, UNIFIED_CLASS_NAMES
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
@register_model("retfound_v61")  # Alias for v6.1 specific configurations
class RETFoundFactory:
    """Factory for RETFound models with v6.1 support"""
    
    # Available pre-trained weights
    PRETRAINED_WEIGHTS = {
        'cfp': {
            'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth',
            'md5': 'a7c8f4e6c3f6b8d9e2f1a5b4c7d8e9f0',
            'description': 'Pre-trained on Color Fundus Photography',
            'modality': 'fundus'
        },
        'oct': {
            'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth',
            'md5': 'b8d9f0e1d2c3a4b5c6d7e8f9a0b1c2d3',
            'description': 'Pre-trained on OCT images',
            'modality': 'oct'
        },
        'meh': {
            'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_meh.pth',
            'md5': 'c9e0f1e2d3c4a5b6c7d8e9f0a1b2c3d4',
            'description': 'Pre-trained on Multi-Ethnic Hybrid dataset',
            'modality': 'both'
        }
    }
    
    # V6.1 specific configurations
    V61_CONFIGS = {
        'fundus': {
            'num_classes': 18,
            'recommended_weights': 'cfp',
            'image_size': 512
        },
        'oct': {
            'num_classes': 10,
            'recommended_weights': 'oct',
            'image_size': 224
        },
        'unified': {
            'num_classes': 28,
            'recommended_weights': 'meh',
            'image_size': 224
        }
    }
    
    @classmethod
    def create(
        cls,
        config: RETFoundConfig,
        pretrained: Optional[str] = None,
        weights_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        unified_classes: bool = None,  # v6.1 specific
        modality: Optional[str] = None,  # v6.1 specific
        **kwargs
    ) -> RETFoundModel:
        """
        Create a RETFound model
        
        Args:
            config: Model configuration
            pretrained: Name of pretrained weights ('cfp', 'oct', 'meh')
            weights_path: Path to custom weights file
            strict: Whether to strictly enforce weight loading
            unified_classes: Whether to use unified 28 classes (v6.1)
            modality: Specific modality ('fundus', 'oct', 'both')
            **kwargs: Additional arguments passed to model
            
        Returns:
            RETFoundModel instance
        """
        # Validate v6.1 specific configurations
        if unified_classes is not None:
            config = cls._validate_v61_config(config, unified_classes, modality)
        
        # Auto-detect v6.1 mode if num_classes is 28
        if config.num_classes == 28 and unified_classes is None:
            unified_classes = True
            logger.info("Auto-detected v6.1 unified classes mode (28 classes)")
        
        # Create model
        logger.info(f"Creating RETFound model: {config.model_type}")
        logger.info(f"Number of classes: {config.num_classes}")
        if unified_classes:
            logger.info("Using v6.1 unified class system")
        
        model = RETFoundModel(
            config, 
            unified_classes=unified_classes,
            modality=modality,
            **kwargs
        )
        
        # Select appropriate pretrained weights for v6.1
        if pretrained is None and unified_classes and modality:
            pretrained = cls._get_recommended_weights(modality)
            logger.info(f"Auto-selected {pretrained} weights for {modality} modality")
        
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
        
        # Log v6.1 specific info
        if unified_classes:
            logger.info(f"V6.1 Configuration:")
            logger.info(f"  - Unified classes: {config.num_classes}")
            logger.info(f"  - Modality: {modality or 'both'}")
            logger.info(f"  - Class names: {len(UNIFIED_CLASS_NAMES)} defined")
        
        return model
    
    @classmethod
    def _validate_v61_config(
        cls,
        config: RETFoundConfig,
        unified_classes: bool,
        modality: Optional[str]
    ) -> RETFoundConfig:
        """Validate and adjust configuration for v6.1"""
        if unified_classes:
            # Ensure correct number of classes
            if config.num_classes != 28:
                logger.warning(
                    f"Adjusting num_classes from {config.num_classes} to 28 "
                    f"for v6.1 unified mode"
                )
                config.num_classes = 28
        else:
            # Modality-specific class counts
            if modality == 'fundus' and config.num_classes != 18:
                logger.warning(
                    f"Adjusting num_classes from {config.num_classes} to 18 "
                    f"for fundus modality"
                )
                config.num_classes = 18
            elif modality == 'oct' and config.num_classes != 10:
                logger.warning(
                    f"Adjusting num_classes from {config.num_classes} to 10 "
                    f"for OCT modality"
                )
                config.num_classes = 10
        
        return config
    
    @classmethod
    def _get_recommended_weights(cls, modality: str) -> str:
        """Get recommended pretrained weights for a modality"""
        if modality in cls.V61_CONFIGS:
            return cls.V61_CONFIGS[modality]['recommended_weights']
        return 'meh'  # Default to multi-ethnic hybrid
    
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
            
            # Log modality compatibility
            weight_modality = cls.PRETRAINED_WEIGHTS[pretrained_type]['modality']
            logger.info(f"Weight modality: {weight_modality}")
            
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
            
            # Check for v6.1 metadata in checkpoint
            if 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                if metadata.get('dataset_version') == 'v6.1':
                    logger.info("Loading v6.1 checkpoint")
                    logger.info(f"  - Classes: {metadata.get('num_classes', 'unknown')}")
                    logger.info(f"  - Modality: {metadata.get('modality', 'unknown')}")
            
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


def list_models() -> List[str]:
    """List all available models"""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model"""
    if model_name not in MODEL_REGISTRY:
        raise ModelNotFoundError(f"Model '{model_name}' not found")
    
    factory = MODEL_REGISTRY.get(model_name)
    
    info = {
        'name': model_name,
        'factory': factory.__name__,
        'pretrained_weights': list(factory.PRETRAINED_WEIGHTS.keys())
        if hasattr(factory, 'PRETRAINED_WEIGHTS') else [],
        'description': factory.__doc__ or "No description available"
    }
    
    # Add v6.1 specific info
    if hasattr(factory, 'V61_CONFIGS'):
        info['v61_support'] = True
        info['v61_modalities'] = list(factory.V61_CONFIGS.keys())
        info['v61_classes'] = {
            k: v['num_classes'] 
            for k, v in factory.V61_CONFIGS.items()
        }
    
    return info


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


# V6.1 specific convenience functions
def create_retfound_v61_unified(
    config: RETFoundConfig,
    pretrained: Optional[str] = 'meh',
    **kwargs
) -> RETFoundModel:
    """Create RETFound model for v6.1 unified 28 classes"""
    config.num_classes = 28  # Ensure correct class count
    return create_model(
        'retfound', 
        config, 
        pretrained=pretrained,
        unified_classes=True,
        modality='both',
        **kwargs
    )


def create_retfound_v61_fundus(
    config: RETFoundConfig,
    pretrained: Optional[str] = 'cfp',
    **kwargs
) -> RETFoundModel:
    """Create RETFound model for v6.1 fundus only (18 classes)"""
    config.num_classes = 18
    return create_model(
        'retfound',
        config,
        pretrained=pretrained,
        unified_classes=False,
        modality='fundus',
        **kwargs
    )


def create_retfound_v61_oct(
    config: RETFoundConfig,
    pretrained: Optional[str] = 'oct',
    **kwargs
) -> RETFoundModel:
    """Create RETFound model for v6.1 OCT only (10 classes)"""
    config.num_classes = 10
    return create_model(
        'retfound',
        config,
        pretrained=pretrained,
        unified_classes=False,
        modality='oct',
        **kwargs
    )


def validate_model_config(
    config: RETFoundConfig,
    dataset_version: str = 'v6.1'
) -> bool:
    """
    Validate model configuration for a specific dataset version
    
    Args:
        config: Model configuration
        dataset_version: Dataset version to validate against
        
    Returns:
        True if valid, raises exception otherwise
    """
    if dataset_version == 'v6.1':
        valid_class_counts = [10, 18, 28]
        if config.num_classes not in valid_class_counts:
            raise ValueError(
                f"Invalid num_classes for v6.1: {config.num_classes}. "
                f"Must be one of {valid_class_counts}"
            )
        
        logger.info(f"Configuration valid for dataset {dataset_version}")
        return True
    
    # Add validation for other dataset versions as needed
    return True


def load_model(
    config: Union[RETFoundConfig, Dict, str],
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = False,
    **kwargs
) -> RETFoundModel:
    """
    Load a model from checkpoint
    
    Args:
        config: Model configuration (config object, dict, or path to config)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce weight loading
        **kwargs: Additional arguments passed to model creation
        
    Returns:
        Loaded RETFoundModel instance
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise WeightsNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config if not provided
    if isinstance(config, str):
        # Config path provided
        from ..core.config import RETFoundConfig
        config = RETFoundConfig.from_file(config)
    elif isinstance(config, dict):
        # Config dict provided
        from ..core.config import RETFoundConfig
        config = RETFoundConfig(**config)
    elif config is None:
        # Try to extract from checkpoint
        if 'config' in checkpoint:
            from ..core.config import RETFoundConfig
            config = RETFoundConfig(**checkpoint['config'])
        else:
            raise ValueError("No config provided and none found in checkpoint")
    
    # Extract metadata for v6.1 compatibility
    metadata = checkpoint.get('metadata', {})
    dataset_version = metadata.get('dataset_version', 'v4.0')
    
    # Auto-detect v6.1 parameters
    unified_classes = None
    modality = None
    
    if dataset_version == 'v6.1':
        unified_classes = metadata.get('unified_classes', True)
        modality = metadata.get('modality', 'both')
        logger.info(f"Loading v6.1 model: unified={unified_classes}, modality={modality}")
    
    # Override with kwargs if provided
    unified_classes = kwargs.pop('unified_classes', unified_classes)
    modality = kwargs.pop('modality', modality)
    
    # Create model
    model = RETFoundFactory.create(
        config,
        unified_classes=unified_classes,
        modality=modality,
        **kwargs
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    
    # Log checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'best_metric' in checkpoint:
        logger.info(f"Best metric: {checkpoint['best_metric']}")
    
    logger.info("Model loaded successfully from checkpoint")
    return model
