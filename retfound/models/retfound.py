"""RETFound Vision Transformer implementation for Dataset v6.1."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
from collections import OrderedDict
from functools import partial
import logging

from timm.models.layers import DropPath, Mlp
from timm.models.vision_transformer import PatchEmbed

from retfound.models.base import BaseModel
from retfound.models.layers import Attention, Block
from retfound.core.exceptions import ModelError
from retfound.core.constants import NUM_TOTAL_CLASSES, UNIFIED_CLASS_NAMES

logger = logging.getLogger(__name__)


class RETFoundModel(BaseModel):
    """RETFound Vision Transformer for medical image classification.
    
    Based on the MAE architecture pre-trained on 1.6M retinal images.
    Updated for Dataset v6.1 with 28 classes (18 Fundus + 10 OCT).
    """
    
    def __init__(self, config: Any):
        """Initialize RETFound model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract model config
        if hasattr(config, 'model'):
            self.img_size = config.data.input_size
            self.patch_size = config.model.patch_size
            self.num_classes = config.model.num_classes
            self.embed_dim = config.model.embed_dim
            self.depth = config.model.depth
            self.num_heads = config.model.num_heads
            self.mlp_ratio = config.model.mlp_ratio
            self.drop_rate = config.model.drop_rate
            self.drop_path_rate = config.model.drop_path_rate
            self.use_gradient_checkpointing = config.optimization.use_gradient_checkpointing
            
            # Dataset v6.1 specific
            self.modality = getattr(config.data, 'modality', 'both')
            self.unified_classes = getattr(config.data, 'unified_classes', True)
        else:
            # Direct config
            self.img_size = getattr(config, 'img_size', 224)
            self.patch_size = getattr(config, 'patch_size', 16)
            self.num_classes = getattr(config, 'num_classes', NUM_TOTAL_CLASSES)  # Default to 28
            self.embed_dim = getattr(config, 'embed_dim', 1024)
            self.depth = getattr(config, 'depth', 24)
            self.num_heads = getattr(config, 'num_heads', 16)
            self.mlp_ratio = getattr(config, 'mlp_ratio', 4.0)
            self.drop_rate = getattr(config, 'drop_rate', 0.0)
            self.drop_path_rate = getattr(config, 'drop_path_rate', 0.2)
            self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
            self.modality = getattr(config, 'modality', 'both')
            self.unified_classes = getattr(config, 'unified_classes', True)
        
        # Validate num_classes for dataset v6.1
        if self.unified_classes and self.num_classes != NUM_TOTAL_CLASSES:
            logger.warning(
                f"num_classes={self.num_classes} but unified_classes=True. "
                f"Setting num_classes to {NUM_TOTAL_CLASSES} for dataset v6.1"
            )
            self.num_classes = NUM_TOTAL_CLASSES
        
        # Build model
        self.build()
        
    def build(self) -> None:
        """Build the RETFound architecture."""
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=self.drop_rate,
                attn_drop=0.,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(self.depth)
        ])
        
        # Final normalization and head
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        
        # Classification head(s) - support for multi-head if needed
        if self.unified_classes:
            # Single head for unified 28 classes
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            # Separate heads for fundus and OCT if not unified
            if self.modality == 'both':
                self.head_fundus = nn.Linear(self.embed_dim, 18)
                self.head_oct = nn.Linear(self.embed_dim, 10)
                self.modality_classifier = nn.Linear(self.embed_dim, 2)
            elif self.modality == 'fundus':
                self.head = nn.Linear(self.embed_dim, 18)
            else:  # oct
                self.head = nn.Linear(self.embed_dim, 10)
        
        # Initialize weights
        self.initialize_weights()
        
        self._is_built = True
        logger.info(f"Built RETFound model for dataset v6.1: {self.summary()}")
    
    def initialize_weights(self) -> None:
        """Initialize model weights following MAE/ViT conventions."""
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize class token and positional embedding
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights_module)
        
    def _init_weights_module(self, m: nn.Module) -> None:
        """Initialize a single module."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_pretrained_weights(
        self,
        checkpoint_path: str,
        model_key: str = 'cfp',
        strict: bool = False
    ) -> Dict[str, Any]:
        """Load RETFound pre-trained weights with proper handling.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_key: Which weights to use ('cfp', 'oct', 'meh')
            strict: Whether to strictly match all keys
            
        Returns:
            Loading information
        """
        logger.info(f"Loading RETFound weights from {checkpoint_path} (key: {model_key})")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Convert MAE weights to our format
        new_state_dict = OrderedDict()
        loaded_keys = set()
        
        for k, v in state_dict.items():
            # Remove prefixes
            orig_k = k
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('encoder.'):
                k = k[8:]
            if k.startswith('backbone.'):
                k = k[9:]
            
            # Skip decoder components (MAE specific)
            if any(skip in k for skip in ['decoder', 'mask_token', 'decoder_pred']):
                continue
            
            # Map keys
            if k in ['cls_token', 'pos_embed']:
                new_state_dict[k] = v
                loaded_keys.add(k)
            elif k.startswith('patch_embed'):
                new_state_dict[k] = v
                loaded_keys.add(k)
            elif k.startswith('blocks'):
                new_state_dict[k] = v
                loaded_keys.add(k)
            elif k in ['norm.weight', 'norm.bias']:
                new_state_dict[k] = v
                loaded_keys.add(k)
            elif 'ln_pre' in k:  # Some checkpoints use ln_pre
                new_k = k.replace('ln_pre', 'norm')
                new_state_dict[new_k] = v
                loaded_keys.add(new_k)
        
        # Handle position embedding interpolation if needed
        if 'pos_embed' in new_state_dict:
            pos_embed_checkpoint = new_state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
            
            # Interpolate if sizes don't match
            if pos_embed_checkpoint.shape[1] != self.pos_embed.shape[1]:
                logger.warning("Position embedding size mismatch, interpolating...")
                pos_embed_checkpoint = self._interpolate_pos_embed(
                    pos_embed_checkpoint, num_patches, num_extra_tokens
                )
                new_state_dict['pos_embed'] = pos_embed_checkpoint
        
        # Load weights
        msg = self.load_state_dict(new_state_dict, strict=strict)
        
        logger.info(f"Loaded {len(loaded_keys)} keys from checkpoint")
        logger.info(f"Missing keys: {len(msg.missing_keys)} (mainly classification head)")
        if msg.unexpected_keys:
            logger.warning(f"Unexpected keys: {len(msg.unexpected_keys)}")
        
        # Initialize classification head(s)
        if hasattr(self, 'head'):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
            logger.info(f"Classification head initialized for {self.num_classes} classes")
        
        if hasattr(self, 'head_fundus'):
            nn.init.trunc_normal_(self.head_fundus.weight, std=0.02)
            nn.init.trunc_normal_(self.head_oct.weight, std=0.02)
            nn.init.trunc_normal_(self.modality_classifier.weight, std=0.02)
            logger.info("Multi-head classification initialized for separate modalities")
        
        return {
            'loaded_keys': list(loaded_keys),
            'missing_keys': msg.missing_keys,
            'unexpected_keys': msg.unexpected_keys,
            'model_key': model_key,
            'num_classes': self.num_classes
        }
    
    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        num_patches: int,
        num_extra_tokens: int
    ) -> torch.Tensor:
        """Interpolate position embeddings to handle different image sizes."""
        # Split position embedding
        extra_tokens = pos_embed[:, :num_extra_tokens]
        pos_tokens = pos_embed[:, num_extra_tokens:]
        
        # Calculate sizes
        orig_size = int((pos_tokens.shape[1]) ** 0.5)
        new_size = int(num_patches ** 0.5)
        
        if orig_size != new_size:
            logger.info(f"Interpolating position embedding from {orig_size}x{orig_size} "
                       f"to {new_size}x{new_size}")
            
            # Reshape to 2D grid
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, pos_embed.shape[-1])
            pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # NCHW
            
            # Interpolate
            pos_tokens = F.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False
            )
            
            # Reshape back
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Concatenate back
        pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        return pos_embed
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, embed_dim]
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Transformer blocks
        if self.training and self.use_gradient_checkpointing:
            # Use gradient checkpointing
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                x = checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        
        # Final norm
        x = self.norm(x)
        
        # Return cls token
        return x[:, 0]
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False,
        modality: Optional[str] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: Whether to return features along with logits
            modality: Modality hint for multi-head models ('fundus' or 'oct')
            
        Returns:
            Logits tensor [B, num_classes] or tuple of (logits, features)
        """
        features = self.forward_features(x)
        
        # Classification based on configuration
        if self.unified_classes or hasattr(self, 'head'):
            # Single unified head
            logits = self.head(features)
        else:
            # Multi-head for separate modalities
            if modality == 'fundus':
                logits = self.head_fundus(features)
            elif modality == 'oct':
                logits = self.head_oct(features)
            else:
                # Need to determine modality
                modality_logits = self.modality_classifier(features)
                modality_probs = F.softmax(modality_logits, dim=1)
                
                # Use predicted modality
                fundus_logits = self.head_fundus(features)
                oct_logits = self.head_oct(features)
                
                # Weighted combination based on modality prediction
                fundus_weight = modality_probs[:, 0:1]
                oct_weight = modality_probs[:, 1:2]
                
                # Pad to same size and combine
                fundus_padded = F.pad(fundus_logits, (0, 10))  # Pad to 28
                oct_padded = F.pad(oct_logits, (18, 0))  # Pad to 28
                
                logits = fundus_weight * fundus_padded + oct_weight * oct_padded
        
        if return_features:
            return logits, features
        return logits
    
    def get_optimizer_params(
        self,
        base_lr: float,
        weight_decay: float = 0.05,
        layer_decay: Optional[float] = 0.65
    ) -> List[Dict[str, Any]]:
        """Get parameter groups with layer-wise learning rate decay.
        
        Args:
            base_lr: Base learning rate
            weight_decay: Weight decay
            layer_decay: Layer-wise decay factor
            
        Returns:
            Parameter groups for optimizer
        """
        if layer_decay is None:
            return super().get_optimizer_params(base_lr, weight_decay)
        
        param_groups = []
        
        # Calculate total depth
        num_layers = len(self.blocks) + 1  # +1 for embedding
        
        # Assign layer indices
        layer_indices = {}
        layer_indices['patch_embed'] = 0
        layer_indices['cls_token'] = 0
        layer_indices['pos_embed'] = 0
        layer_indices['pos_drop'] = 0
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            layer_indices[f'blocks.{i}'] = i + 1
        
        # Final layers
        layer_indices['norm'] = num_layers
        layer_indices['head'] = num_layers + 1
        layer_indices['head_fundus'] = num_layers + 1
        layer_indices['head_oct'] = num_layers + 1
        layer_indices['modality_classifier'] = num_layers + 1
        
        # Group parameters
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Find layer index
            layer_id = num_layers + 1  # Default to last layer
            for key in layer_indices:
                if key in name:
                    layer_id = layer_indices[key]
                    break
            
            # Calculate learning rate scale
            if any(head in name for head in ['head', 'classifier']):
                # Classification heads get full learning rate
                lr_scale = 1.0
            else:
                lr_scale = layer_decay ** (num_layers - layer_id)
            
            # Determine weight decay
            if weight_decay > 0:
                if 'bias' in name or 'norm' in name:
                    group_weight_decay = 0.0
                else:
                    group_weight_decay = weight_decay
            else:
                group_weight_decay = 0.0
            
            param_groups.append({
                'params': [param],
                'lr': base_lr * lr_scale,
                'weight_decay': group_weight_decay,
                'param_name': name,
                'lr_scale': lr_scale,
                'layer_id': layer_id
            })
        
        # Log parameter groups info
        logger.info(f"Created {len(param_groups)} parameter groups with layer-wise LR decay")
        
        return param_groups
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory."""
        self.use_gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled")
    
    def get_layer_groups(self, num_groups: int = 4) -> List[List[nn.Module]]:
        """Get layer groups for differential learning rates.
        
        Args:
            num_groups: Number of groups to create
            
        Returns:
            List of layer groups
        """
        # Group 0: Embeddings
        group0 = [self.patch_embed, self.cls_token, self.pos_embed]
        
        # Divide blocks into groups
        blocks_per_group = len(self.blocks) // (num_groups - 2)
        block_groups = []
        
        for i in range(num_groups - 2):
            start_idx = i * blocks_per_group
            if i == num_groups - 3:
                # Last block group gets remaining blocks
                block_groups.append(self.blocks[start_idx:])
            else:
                end_idx = start_idx + blocks_per_group
                block_groups.append(self.blocks[start_idx:end_idx])
        
        # Last group: Head(s)
        final_group = [self.norm]
        if hasattr(self, 'head'):
            final_group.append(self.head)
        if hasattr(self, 'head_fundus'):
            final_group.extend([self.head_fundus, self.head_oct, self.modality_classifier])
        
        groups = [group0] + block_groups + [final_group]
        return groups
    
    def get_class_names(self) -> List[str]:
        """Get class names for the model.
        
        Returns:
            List of class names
        """
        if self.unified_classes:
            return UNIFIED_CLASS_NAMES
        else:
            # Return appropriate class names based on modality
            if self.modality == 'fundus':
                from retfound.core.constants import FUNDUS_CLASS_NAMES
                return FUNDUS_CLASS_NAMES
            elif self.modality == 'oct':
                from retfound.core.constants import OCT_CLASS_NAMES
                return OCT_CLASS_NAMES
            else:
                # Combined
                return UNIFIED_CLASS_NAMES
    
    def summary(self) -> str:
        """Get model summary.
        
        Returns:
            Model summary string
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"RETFoundModel(\n"
        summary += f"  img_size={self.img_size}, patch_size={self.patch_size}\n"
        summary += f"  embed_dim={self.embed_dim}, depth={self.depth}, num_heads={self.num_heads}\n"
        summary += f"  num_classes={self.num_classes}, modality={self.modality}\n"
        summary += f"  unified_classes={self.unified_classes}\n"
        summary += f"  total_params={total_params:,}, trainable_params={trainable_params:,}\n"
        summary += f")"
        
        return summary