"""Model architectures for RETFound Training Framework."""

from retfound.models.base import BaseModel
from retfound.models.retfound import RETFoundModel
from retfound.models.factory import create_model, load_pretrained_weights

__all__ = [
    "BaseModel",
    "RETFoundModel",
    "create_model",
    "load_pretrained_weights",
]

# Register default models
from retfound.core.registry import MODEL_REGISTRY

# Register RETFound models
MODEL_REGISTRY.register(
    "retfound_vit_large",
    RETFoundModel,
    default_config={
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
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
    }
)

# Alias for backward compatibility
MODEL_REGISTRY.register("vit_large_patch16_224", RETFoundModel)