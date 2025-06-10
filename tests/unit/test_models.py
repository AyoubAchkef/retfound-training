"""
Model Tests
===========

Test model architecture, forward pass, and weight loading.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from retfound.core.config import RETFoundConfig
from retfound.models.retfound import RETFoundModel, Attention, Block
from retfound.models.factory import create_model, MODEL_REGISTRY


class TestRETFoundModel:
    """Test RETFound model architecture"""
    
    def test_model_creation(self, minimal_config):
        """Test basic model creation"""
        model = RETFoundModel(minimal_config)
        
        # Check model attributes
        assert hasattr(model, 'patch_embed')
        assert hasattr(model, 'cls_token')
        assert hasattr(model, 'pos_embed')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'head')
        
        # Check dimensions
        assert model.cls_token.shape == (1, 1, minimal_config.embed_dim)
        assert len(model.blocks) == minimal_config.depth
    
    def test_forward_pass(self, sample_model, sample_batch):
        """Test model forward pass"""
        model = sample_model
        images, labels = sample_batch
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(images)
        
        # Check output shape
        batch_size = images.shape[0]
        num_classes = model.config.num_classes
        assert output.shape == (batch_size, num_classes)
        
        # Check output is valid
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, sample_model, sample_batch):
        """Test gradient flow through model"""
        model = sample_model
        images, labels = sample_batch
        
        # Enable gradients
        model.train()
        images.requires_grad_(True)
        
        # Forward pass
        output = model(images)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_patch_embedding(self, minimal_config):
        """Test patch embedding layer"""
        model = RETFoundModel(minimal_config)
        
        # Create input
        batch_size = 2
        images = torch.randn(batch_size, 3, minimal_config.input_size, minimal_config.input_size)
        
        # Get patch embeddings
        patches = model.patch_embed(images)
        
        # Calculate expected dimensions
        num_patches = (minimal_config.input_size // minimal_config.patch_size) ** 2
        
        # Check shape
        assert patches.shape == (batch_size, num_patches, minimal_config.embed_dim)
    
    def test_position_embedding_interpolation(self, minimal_config):
        """Test position embedding interpolation for different sizes"""
        # Create model with standard size
        model = RETFoundModel(minimal_config)
        
        # Create position embeddings for different size
        original_pos_embed = torch.randn(1, 197, minimal_config.embed_dim)  # 196 patches + 1 cls
        
        # Test interpolation
        new_size = 16  # Results in 256 patches
        num_patches = new_size * new_size
        interpolated = model._interpolate_pos_embed(original_pos_embed, num_patches, 1)
        
        # Check shape
        assert interpolated.shape == (1, num_patches + 1, minimal_config.embed_dim)
    
    def test_load_pretrained_weights(self, minimal_config, sample_weights_file):
        """Test loading pretrained weights"""
        model = RETFoundModel(minimal_config)
        
        # Load weights
        msg = model.load_pretrained_weights(sample_weights_file, model_key='test')
        
        # Check some weights were loaded
        assert len(msg.missing_keys) > 0  # Head should be missing
        assert 'head.weight' in msg.missing_keys
        assert 'head.bias' in msg.missing_keys
    
    def test_gradient_checkpointing(self, minimal_config, sample_batch):
        """Test gradient checkpointing functionality"""
        minimal_config.use_gradient_checkpointing = True
        model = RETFoundModel(minimal_config)
        
        images, labels = sample_batch
        
        # Training mode enables checkpointing
        model.train()
        output = model(images)
        loss = output.sum()
        loss.backward()
        
        # Should complete without error
        assert True


class TestAttentionBlock:
    """Test attention mechanism"""
    
    def test_attention_creation(self):
        """Test attention block creation"""
        dim = 768
        num_heads = 12
        attn = Attention(dim=dim, num_heads=num_heads)
        
        # Check components
        assert hasattr(attn, 'qkv')
        assert hasattr(attn, 'proj')
        assert attn.num_heads == num_heads
        assert attn.scale == (dim // num_heads) ** -0.5
    
    def test_attention_forward(self):
        """Test attention forward pass"""
        batch_size = 2
        seq_len = 197
        dim = 768
        
        attn = Attention(dim=dim, num_heads=12)
        x = torch.randn(batch_size, seq_len, dim)
        
        # Forward pass
        output = attn(x)
        
        # Check output
        assert output.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(output).any()


class TestTransformerBlock:
    """Test transformer block"""
    
    def test_block_creation(self):
        """Test transformer block creation"""
        dim = 768
        block = Block(
            dim=dim,
            num_heads=12,
            mlp_ratio=4.0,
            drop_path=0.1
        )
        
        # Check components
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'attn')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'drop_path')
    
    def test_block_forward(self):
        """Test transformer block forward"""
        batch_size = 2
        seq_len = 197
        dim = 768
        
        block = Block(dim=dim, num_heads=12)
        x = torch.randn(batch_size, seq_len, dim)
        
        # Forward pass
        output = block(x)
        
        # Check output
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        
        # Check residual connection
        # Output should be different from input
        assert not torch.allclose(output, x)


class TestModelFactory:
    """Test model factory and registry"""
    
    def test_model_registration(self):
        """Test model registration in factory"""
        # Check RETFound is registered
        assert 'retfound' in MODEL_REGISTRY
        
        # Register a dummy model
        @MODEL_REGISTRY.register('dummy')
        class DummyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.fc = nn.Linear(10, config.num_classes)
            
            def forward(self, x):
                return self.fc(x)
        
        # Check it's registered
        assert 'dummy' in MODEL_REGISTRY
    
    def test_create_model(self, minimal_config):
        """Test model creation through factory"""
        # Create RETFound model
        model = create_model(minimal_config)
        
        assert isinstance(model, RETFoundModel)
        assert model.config == minimal_config
    
    def test_create_unknown_model(self, minimal_config):
        """Test creating unknown model raises error"""
        minimal_config.model_type = 'unknown_model'
        
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(minimal_config)
    
    def test_list_models(self):
        """Test listing available models"""
        models = MODEL_REGISTRY.list_models()
        
        assert isinstance(models, list)
        assert 'retfound' in models


class TestModelUtils:
    """Test model utility functions"""
    
    def test_count_parameters(self, sample_model):
        """Test parameter counting"""
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params trainable by default
    
    def test_freeze_layers(self, sample_model):
        """Test freezing model layers"""
        # Freeze patch embedding
        for param in sample_model.patch_embed.parameters():
            param.requires_grad = False
        
        # Count trainable params
        trainable_before = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in sample_model.parameters())
        
        assert trainable_before < total_params
        
        # Unfreeze
        for param in sample_model.patch_embed.parameters():
            param.requires_grad = True
        
        trainable_after = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        assert trainable_after == total_params
    
    def test_model_device_movement(self, sample_model, sample_batch):
        """Test moving model between devices"""
        images, labels = sample_batch
        
        # CPU inference
        sample_model.cpu()
        sample_model.eval()
        with torch.no_grad():
            output_cpu = sample_model(images.cpu())
        
        # Move to GPU if available
        if torch.cuda.is_available():
            sample_model.cuda()
            with torch.no_grad():
                output_gpu = sample_model(images.cuda())
            
            # Results should be very close
            assert torch.allclose(output_cpu, output_gpu.cpu(), rtol=1e-5)
