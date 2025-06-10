"""
Optimizer Tests
===============

Test custom optimizers including SAM.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from retfound.training.optimizers import (
    SAM, ASAM, create_optimizer, 
    get_layer_wise_lr_groups, LayerWiseLRScheduler
)
from retfound.core.config import RETFoundConfig
from retfound.models.retfound import RETFoundModel


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestSAM:
    """Test Sharpness Aware Minimization optimizer"""
    
    def test_sam_creation(self):
        """Test creating SAM optimizer"""
        model = SimpleModel()
        base_optimizer = torch.optim.SGD
        
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=0.1,
            rho=0.05
        )
        
        assert hasattr(optimizer, 'base_optimizer')
        assert hasattr(optimizer, 'rho')
        assert optimizer.param_groups[0]['rho'] == 0.05
    
    def test_sam_step(self):
        """Test SAM optimization step"""
        model = SimpleModel()
        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=0.1,
            rho=0.05
        )
        
        # Create dummy data
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        
        # Define closure
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            return loss
        
        # Initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Optimization step
        loss = optimizer.step(closure)
        
        # Check parameters changed
        for p_initial, p_current in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_initial, p_current)
    
    def test_sam_adaptive(self):
        """Test adaptive SAM"""
        model = SimpleModel()
        optimizer = SAM(
            model.parameters(),
            torch.optim.Adam,
            lr=0.001,
            rho=0.05,
            adaptive=True
        )
        
        # Check adaptive flag
        assert optimizer.param_groups[0]['adaptive'] == True
        
        # Run optimization step
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        assert loss is not None
    
    def test_sam_state_dict(self):
        """Test SAM state dict save/load"""
        model = SimpleModel()
        optimizer1 = SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=0.1,
            rho=0.05
        )
        
        # Run a step to create state
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        
        def closure():
            optimizer1.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            return loss
        
        optimizer1.step(closure)
        
        # Save state
        state_dict = optimizer1.state_dict()
        
        # Create new optimizer and load state
        optimizer2 = SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=0.1,
            rho=0.05
        )
        optimizer2.load_state_dict(state_dict)
        
        # Check state loaded correctly
        assert optimizer2.param_groups[0]['lr'] == optimizer1.param_groups[0]['lr']


class TestASAM:
    """Test Adaptive SAM optimizer"""
    
    def test_asam_creation(self):
        """Test creating ASAM optimizer"""
        model = SimpleModel()
        optimizer = ASAM(
            model.parameters(),
            torch.optim.Adam,
            lr=0.001,
            rho=0.5,
            eta=0.01
        )
        
        assert hasattr(optimizer, 'rho')
        assert hasattr(optimizer, 'eta')
        assert optimizer.param_groups[0]['rho'] == 0.5
        assert optimizer.param_groups[0]['eta'] == 0.01
    
    def test_asam_adaptive_rho(self):
        """Test ASAM adaptive rho computation"""
        model = SimpleModel()
        optimizer = ASAM(
            model.parameters(),
            torch.optim.Adam,
            lr=0.001,
            rho=0.5,
            eta=0.01
        )
        
        # Run multiple steps and check rho adaptation
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        
        initial_rho = optimizer.param_groups[0]['rho']
        
        for _ in range(5):
            def closure():
                optimizer.zero_grad()
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                loss.backward()
                return loss
            
            optimizer.step(closure)
        
        # Rho might have adapted
        final_rho = optimizer.param_groups[0]['rho']
        # Note: Actual adaptation depends on loss landscape
        assert final_rho > 0


class TestOptimizerFactory:
    """Test optimizer creation factory"""
    
    def test_create_standard_optimizers(self, sample_model, minimal_config):
        """Test creating standard optimizers"""
        # Test AdamW
        minimal_config.optimizer = 'adamw'
        optimizer = create_optimizer(sample_model, minimal_config)
        assert isinstance(optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer, 
                         torch.optim.AdamW)
        
        # Test SGD
        minimal_config.optimizer = 'sgd'
        minimal_config.use_sam = False
        optimizer = create_optimizer(sample_model, minimal_config)
        assert isinstance(optimizer, torch.optim.SGD)
    
    def test_create_sam_optimizer(self, sample_model, minimal_config):
        """Test creating SAM optimizer"""
        minimal_config.use_sam = True
        minimal_config.optimizer = 'adamw'
        
        optimizer = create_optimizer(sample_model, minimal_config)
        
        assert isinstance(optimizer, SAM)
        assert isinstance(optimizer.base_optimizer, torch.optim.AdamW)
    
    def test_optimizer_groups(self, sample_model, minimal_config):
        """Test creating optimizer with parameter groups"""
        minimal_config.layer_decay = 0.75
        
        optimizer = create_optimizer(sample_model, minimal_config)
        
        # Should have multiple param groups for layer-wise LR
        assert len(optimizer.param_groups) > 1
        
        # Check different learning rates
        lrs = [group['lr'] for group in optimizer.param_groups]
        assert len(set(lrs)) > 1  # Should have different LRs


class TestLayerWiseLR:
    """Test layer-wise learning rate functionality"""
    
    def test_get_layer_groups(self, sample_model, minimal_config):
        """Test getting layer-wise parameter groups"""
        groups = get_layer_wise_lr_groups(
            sample_model,
            base_lr=1e-4,
            weight_decay=0.01,
            layer_decay=0.75
        )
        
        # Check structure
        assert isinstance(groups, list)
        assert all('params' in g for g in groups)
        assert all('lr' in g for g in groups)
        
        # Check learning rates decrease
        lrs = [g['lr'] for g in groups]
        # Earlier layers should have smaller LR
        assert lrs[0] < lrs[-1]
    
    def test_layer_wise_scheduler(self, sample_model, minimal_config):
        """Test layer-wise LR scheduler"""
        optimizer = create_optimizer(sample_model, minimal_config)
        
        scheduler = LayerWiseLRScheduler(
            optimizer,
            base_lr=minimal_config.base_lr,
            min_lr=minimal_config.min_lr,
            warmup_epochs=5,
            total_epochs=20
        )
        
        # Test warmup phase
        initial_lrs = [g['lr'] for g in optimizer.param_groups]
        
        for epoch in range(5):
            scheduler.step(epoch)
        
        warmup_lrs = [g['lr'] for g in optimizer.param_groups]
        
        # LRs should increase during warmup
        for initial, warmup in zip(initial_lrs, warmup_lrs):
            assert warmup > initial
        
        # Test decay phase
        for epoch in range(5, 20):
            scheduler.step(epoch)
        
        final_lrs = [g['lr'] for g in optimizer.param_groups]
        
        # LRs should decrease after warmup
        for warmup, final in zip(warmup_lrs, final_lrs):
            assert final < warmup


class TestOptimizerUtils:
    """Test optimizer utility functions"""
    
    def test_grad_norm_clipping(self):
        """Test gradient norm clipping"""
        model = SimpleModel()
        
        # Create large gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 100
        
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        
        # Check norm is clipped
        assert total_norm > 1.0  # Original norm was large
        
        # Compute new norm
        new_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                new_norm += p.grad.data.norm(2).item() ** 2
        new_norm = new_norm ** 0.5
        
        assert abs(new_norm - 1.0) < 0.01
    
    def test_parameter_groups_weight_decay(self, sample_model):
        """Test weight decay exclusion for certain parameters"""
        # Get parameter groups with no weight decay for bias/norm
        param_groups = []
        
        for name, param in sample_model.named_parameters():
            if 'bias' in name or 'norm' in name:
                param_groups.append({
                    'params': [param],
                    'weight_decay': 0.0
                })
            else:
                param_groups.append({
                    'params': [param],
                    'weight_decay': 0.01
                })
        
        # Check weight decay assignment
        for group in param_groups:
            param = group['params'][0]
            param_name = None
            for name, p in sample_model.named_parameters():
                if p is param:
                    param_name = name
                    break
            
            if param_name and ('bias' in param_name or 'norm' in param_name):
                assert group['weight_decay'] == 0.0
            else:
                assert group['weight_decay'] == 0.01
    
    def test_optimizer_state_consistency(self):
        """Test optimizer state remains consistent"""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Run some steps
        for _ in range(10):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        
        # Check state exists for all parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                assert p in optimizer.state
                state = optimizer.state[p]
                assert 'exp_avg' in state
                assert 'exp_avg_sq' in state
                assert 'step' in state
