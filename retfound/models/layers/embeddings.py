"""Embedding layers for RETFound models."""

import torch
import torch.nn as nn
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer.
    
    Converts input images into patch embeddings by applying a convolutional layer.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        """Initialize patch embedding.
        
        Args:
            image_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            bias: Whether to use bias in the projection layer
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch projection using convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Check input dimensions
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}x{W}) doesn't match model ({self.image_size}x{self.image_size})"
        
        # Apply patch projection: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to get (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class PositionalEmbedding(nn.Module):
    """Positional embedding layer for Vision Transformer.
    
    Adds learnable positional embeddings to patch embeddings.
    """
    
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        dropout: float = 0.0,
        class_token: bool = True,
    ):
        """Initialize positional embedding.
        
        Args:
            num_patches: Number of patches
            embed_dim: Embedding dimension
            dropout: Dropout rate
            class_token: Whether to include class token
        """
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.class_token = class_token
        
        # Class token (if used)
        if class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_positions = num_patches + 1  # +1 for class token
        else:
            self.cls_token = None
            num_positions = num_patches
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize class token
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Patch embeddings of shape (B, num_patches, embed_dim)
            
        Returns:
            Embeddings with positional encoding of shape (B, seq_len, embed_dim)
        """
        B, N, D = x.shape
        
        # Add class token if used
        if self.class_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (fixed, not learnable).
    
    Alternative to learnable positional embeddings using sinusoidal functions.
    """
    
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        dropout: float = 0.0,
        class_token: bool = True,
        temperature: float = 10000.0,
    ):
        """Initialize sinusoidal positional embedding.
        
        Args:
            num_patches: Number of patches
            embed_dim: Embedding dimension
            dropout: Dropout rate
            class_token: Whether to include class token
            temperature: Temperature for sinusoidal encoding
        """
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.class_token = class_token
        self.temperature = temperature
        
        # Class token (if used)
        if class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_positions = num_patches + 1
        else:
            self.cls_token = None
            num_positions = num_patches
        
        # Create sinusoidal positional embeddings
        pos_embed = self._create_sinusoidal_embeddings(num_positions, embed_dim)
        self.register_buffer('pos_embed', pos_embed)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize class token
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def _create_sinusoidal_embeddings(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        """Create sinusoidal positional embeddings.
        
        Args:
            num_positions: Number of positions
            embed_dim: Embedding dimension
            
        Returns:
            Sinusoidal embeddings of shape (1, num_positions, embed_dim)
        """
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            -(math.log(self.temperature) / embed_dim)
        )
        
        pos_embed = torch.zeros(num_positions, embed_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embed.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Patch embeddings of shape (B, num_patches, embed_dim)
            
        Returns:
            Embeddings with positional encoding of shape (B, seq_len, embed_dim)
        """
        B, N, D = x.shape
        
        # Add class token if used
        if self.class_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class HybridEmbedding(nn.Module):
    """Hybrid embedding that combines CNN features with patch embedding.
    
    Uses a CNN backbone to extract features before patch embedding.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        image_size: int = 224,
        feature_size: Optional[int] = None,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """Initialize hybrid embedding.
        
        Args:
            backbone: CNN backbone for feature extraction
            image_size: Input image size
            feature_size: Size of feature maps from backbone (auto-computed if None)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        # Compute feature size if not provided
        if feature_size is None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, image_size, image_size)
                features = backbone(dummy_input)
                feature_size = features.shape[-1]
        
        self.feature_size = feature_size
        self.num_patches = feature_size ** 2
        
        # Projection layer to embed_dim
        self.proj = nn.Conv2d(
            backbone.num_features if hasattr(backbone, 'num_features') else features.shape[1],
            embed_dim,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # Extract features using backbone
        x = self.backbone(x)
        
        # Project to embedding dimension
        x = self.proj(x)
        
        # Flatten spatial dimensions and transpose
        x = x.flatten(2).transpose(1, 2)
        
        return x
