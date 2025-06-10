"""Transformer blocks for Vision Transformer architectures."""

import torch
import torch.nn as nn
from typing import Optional, Callable, Union
from functools import partial

from timm.models.layers import DropPath, Mlp
from retfound.models.layers.attention import Attention


class Block(nn.Module):
    """Transformer block with attention and MLP.
    
    This is the basic building block of Vision Transformers.
    Includes:
    - Multi-head self-attention
    - MLP with GELU activation
    - Residual connections
    - Layer normalization
    - Stochastic depth (drop path)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        init_values: Optional[float] = None,
    ):
        """Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer class
            norm_layer: Normalization layer class
            init_values: Initial values for layer scale
        """
        super().__init__()
        
        # Normalization layers
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Drop path (stochastic depth)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        
        # Layer scale (if specified)
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape [B, N, C]
            
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Attention block
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x)))
        
        # MLP block
        if self.gamma_2 is None:
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x


class ParallelBlock(nn.Module):
    """Parallel transformer block where attention and MLP run in parallel.
    
    This is an experimental design that can be more efficient.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
    ):
        """Initialize parallel transformer block."""
        super().__init__()
        
        # Single normalization for input
        self.norm = norm_layer(dim)
        
        # Attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Learnable mixing parameter
        self.mixing = nn.Parameter(torch.ones(2, 1, 1, dim) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of parallel transformer block.
        
        Args:
            x: Input tensor of shape [B, N, C]
            
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Normalize input
        normed = self.norm(x)
        
        # Run attention and MLP in parallel
        attn_out = self.attn(normed)
        mlp_out = self.mlp(normed)
        
        # Mix outputs
        mixed = self.mixing[0] * attn_out + self.mixing[1] * mlp_out
        
        # Residual connection with drop path
        x = x + self.drop_path(mixed)
        
        return x


class ConvBlock(nn.Module):
    """Convolutional block that can be mixed with transformer blocks.
    
    Useful for hybrid architectures.
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        drop: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.BatchNorm2d,
    ):
        """Initialize convolutional block."""
        super().__init__()
        
        self.conv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=groups,
            bias=bias
        )
        self.norm = norm_layer(dim)
        self.act = act_layer()
        self.drop = nn.Dropout2d(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of convolutional block.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        identity = x
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        
        x = x + identity
        
        return x