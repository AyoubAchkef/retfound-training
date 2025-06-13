"""Model architectures for RETFound Training Framework - Enhanced for v6.1."""

from retfound.models.base import BaseModel
from retfound.models.retfound import RETFoundModel
from retfound.models.factory import (
    create_model,
    load_model,
    create_retfound_cfp,
    create_retfound_oct,
    create_retfound_meh,
    create_retfound_v61_unified,
    create_retfound_v61_fundus,
    create_retfound_v61_oct,
    list_models,
    get_model_info,
    validate_model_config,
    RETFoundFactory
)

__all__ = [
    # Base classes
    "BaseModel",
    "RETFoundModel",
    
    # Factory functions
    "create_model",
    "load_model",
    "create_retfound_cfp",
    "create_retfound_oct",
    "create_retfound_meh",
    
    # V6.1 specific functions
    "create_retfound_v61_unified",
    "create_retfound_v61_fundus",
    "create_retfound_v61_oct",
    
    # Utility functions
    "list_models",
    "get_model_info",
    "validate_model_config",
    
    # Factory class
    "RETFoundFactory",
]

# Register default models
from retfound.core.registry import MODEL_REGISTRY

# Register standard RETFound models
MODEL_REGISTRY.register(
    "retfound_vit_large",
    RETFoundModel,
    default_config={
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "num_classes": 22,  # Default for backward compatibility
    }
)

MODEL_REGISTRY.register(
    "retfound_vit_base",
    RETFoundModel,
    default_config={
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
        "num_classes": 22,  # Default for backward compatibility
    }
)

# Register v6.1 specific models
MODEL_REGISTRY.register(
    "retfound_v61_unified",
    RETFoundModel,
    default_config={
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "num_classes": 28,  # Unified 28 classes
        "unified_classes": True,
        "modality": "both",
    }
)

MODEL_REGISTRY.register(
    "retfound_v61_fundus",
    RETFoundModel,
    default_config={
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "num_classes": 18,  # Fundus only
        "unified_classes": False,
        "modality": "fundus",
        "image_size": 512,  # Larger for fundus
    }
)

MODEL_REGISTRY.register(
    "retfound_v61_oct",
    RETFoundModel,
    default_config={
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "num_classes": 10,  # OCT only
        "unified_classes": False,
        "modality": "oct",
        "image_size": 224,  # Standard for OCT
    }
)

# Register multi-scale variants for v6.1
MODEL_REGISTRY.register(
    "retfound_v61_multiscale",
    RETFoundModel,
    default_config={
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "num_classes": 28,
        "unified_classes": True,
        "modality": "both",
        "enable_multiscale": True,  # For multi-resolution training
    }
)

# Alias for backward compatibility
MODEL_REGISTRY.register("vit_large_patch16_224", RETFoundModel)

# Register factory itself for direct access
MODEL_REGISTRY.register("retfound_factory", RETFoundFactory)


# Convenience function to get v6.1 model configurations
def get_v61_model_config(modality: str = "unified") -> dict:
    """Get recommended configuration for v6.1 models.
    
    Args:
        modality: One of 'unified', 'fundus', 'oct'
        
    Returns:
        Dictionary with model configuration
    """
    configs = {
        "unified": {
            "model_name": "retfound_v61_unified",
            "num_classes": 28,
            "pretrained": "meh",
            "image_size": 224,
        },
        "fundus": {
            "model_name": "retfound_v61_fundus",
            "num_classes": 18,
            "pretrained": "cfp",
            "image_size": 512,
        },
        "oct": {
            "model_name": "retfound_v61_oct",
            "num_classes": 10,
            "pretrained": "oct",
            "image_size": 224,
        },
    }
    
    if modality not in configs:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(configs.keys())}")
    
    return configs[modality]


# Function to check if a model supports v6.1
def is_v61_compatible(model_name: str) -> bool:
    """Check if a model is compatible with dataset v6.1.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model supports v6.1
    """
    v61_models = [
        "retfound_v61_unified",
        "retfound_v61_fundus", 
        "retfound_v61_oct",
        "retfound_v61_multiscale",
    ]
    return model_name in v61_models or model_name.startswith("retfound")
