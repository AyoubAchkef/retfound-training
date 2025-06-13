"""
Optimizer Classes for RETFound
=============================

Implements advanced optimizers including SAM (Sharpness Aware Minimization)
and EMA (Exponential Moving Average).
"""

import logging
import copy
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..core.config import RETFoundConfig
from ..core.registry import Registry

logger = logging.getLogger(__name__)

# Optimizer registry
OPTIMIZER_REGISTRY = Registry("optimizers")


def register_optimizer(name: str):
    """Decorator to register an optimizer"""
    def decorator(cls):
        OPTIMIZER_REGISTRY.register(name, cls)
        return cls
    return decorator


class SAM(Optimizer):
    """
    Sharpness Aware Minimization (SAM) optimizer
    
    Reference: https://arxiv.org/abs/2010.01412
    """
    
    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs
    ):
        """
        Initialize SAM optimizer
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
            rho: Neighborhood size
            adaptive: Whether to use adaptive SAM
            **kwargs: Arguments for base optimizer
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute ε(w) and update weights to w + ε(w)"""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Save original parameters
                self.state[p]["old_p"] = p.data.clone()
                
                # Compute ε(w)
                if group["adaptive"]:
                    # Adaptive SAM: scale by parameter magnitude
                    e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                else:
                    # Original SAM
                    e_w = p.grad * scale.to(p)
                
                # Update to w + ε(w)
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Update weights using gradients at w + ε(w)"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original parameters
                p.data = self.state[p]["old_p"]
        
        # Apply base optimizer step
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure"
        
        # Enable gradient computation for closure
        closure = torch.enable_grad()(closure)
        
        # First forward-backward pass
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        closure()
        
        # Update weights
        self.second_step()
    
    def _grad_norm(self) -> torch.Tensor:
        """Compute gradient norm"""
        shared_device = self.param_groups[0]["params"][0].device
        
        # Compute norm of all gradients
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        
        return norm
    
    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class EMA:
    """
    Exponential Moving Average of model parameters
    
    Maintains a moving average of model parameters for more stable predictions
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA
        
        Args:
            model: Model to track
            decay: Decay rate for moving average
            update_after_step: Start updating after this many steps
            update_every: Update frequency
            device: Device for EMA model
        """
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.device = device or next(model.parameters()).device
        
        # Create EMA model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        
        # Move to device
        self.ema_model.to(self.device)
        
        # Step counter
        self.step = 0
    
    @torch.no_grad()
    def update(self):
        """Update EMA parameters"""
        self.step += 1
        
        # Check if we should update
        if self.step < self.update_after_step:
            return
        
        if self.step % self.update_every != 0:
            return
        
        # Update EMA parameters
        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            self.model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1 - self.decay
            )
        
        # Update buffers (batch norm stats, etc.)
        for ema_buffer, model_buffer in zip(
            self.ema_model.buffers(),
            self.model.buffers()
        ):
            ema_buffer.data.copy_(model_buffer.data)
    
    def forward(self, *args, **kwargs):
        """Forward pass through EMA model"""
        return self.ema_model(*args, **kwargs)
    
    def eval(self):
        """Set EMA model to eval mode"""
        self.ema_model.eval()
    
    def train(self):
        """Set EMA model to train mode"""
        self.ema_model.train()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state dict"""
        return {
            'step': self.step,
            'decay': self.decay,
            'ema_model_state_dict': self.ema_model.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state dict"""
        self.step = state_dict.get('step', 0)
        self.decay = state_dict.get('decay', self.decay)
        self.ema_model.load_state_dict(state_dict['ema_model_state_dict'])


def get_parameter_groups(
    model: nn.Module,
    config: RETFoundConfig
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning rate decay
    
    Args:
        model: Model to get parameters from
        config: Configuration object
        
    Returns:
        List of parameter groups
    """
    param_groups = []
    lr_scales = {}
    
    # Special handling for vision transformers
    if hasattr(model, 'blocks'):
        # Calculate total depth
        num_layers = len(model.blocks) + 1  # +1 for embedding
        
        # Assign layer indices
        layer_indices = {}
        
        # Embedding layers get layer_id 0
        for name in ['patch_embed', 'cls_token', 'pos_embed', 'pos_drop']:
            layer_indices[name] = 0
        
        # Transformer blocks
        for i in range(len(model.blocks)):
            layer_indices[f'blocks.{i}'] = i + 1
        
        # Final layers
        layer_indices['norm'] = num_layers
        layer_indices['head'] = num_layers + 1
        
        # Group parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Find layer index
            layer_id = num_layers + 1  # Default to last layer
            for key in layer_indices:
                if key in name:
                    layer_id = layer_indices[key]
                    break
            
            # Calculate learning rate scale
            if 'head' in name:
                # Classification head gets full learning rate
                lr_scale = 1.0
            else:
                # Layer decay
                lr_scale = config.layer_decay ** (num_layers - layer_id)
            
            lr_scales[name] = lr_scale
            
            # Determine weight decay
            if config.weight_decay > 0:
                # No weight decay for bias and normalization parameters
                if 'bias' in name or 'norm' in name:
                    group_weight_decay = 0.0
                else:
                    group_weight_decay = config.weight_decay
            else:
                group_weight_decay = 0.0
            
            param_groups.append({
                'params': [param],
                'lr': config.base_lr * lr_scale,
                'weight_decay': group_weight_decay,
                'param_name': name,
                'lr_scale': lr_scale,
                'layer_id': layer_id
            })
    
    else:
        # Standard parameter grouping (no layer decay)
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    
    # Log parameter groups
    logger.info(f"Created {len(param_groups)} parameter groups")
    
    if lr_scales:
        # Log layer-wise LR info
        logger.info("Layer-wise Learning Rate Decay:")
        layer_groups = defaultdict(list)
        for name, scale in lr_scales.items():
            for key, layer_id in layer_indices.items():
                if key in name:
                    layer_groups[layer_id].append((name, scale))
                    break
        
        for layer_id in sorted(layer_groups.keys()):
            if layer_groups[layer_id]:
                name_example = layer_groups[layer_id][0][0]
                lr_scale = layer_groups[layer_id][0][1]
                effective_lr = config.base_lr * lr_scale
                logger.info(
                    f"Layer {layer_id:2d}: LR scale={lr_scale:.4f}, "
                    f"Effective LR={effective_lr:.2e} (e.g., {name_example})"
                )
    
    return param_groups


@register_optimizer("adamw")
def create_adamw(
    model: nn.Module,
    config: RETFoundConfig,
    **kwargs
) -> torch.optim.AdamW:
    """Create AdamW optimizer"""
    param_groups = get_parameter_groups(model, config)
    
    return torch.optim.AdamW(
        param_groups,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
        **kwargs
    )


@register_optimizer("sam_adamw")
def create_sam_adamw(
    model: nn.Module,
    config: RETFoundConfig,
    **kwargs
) -> SAM:
    """Create SAM optimizer with AdamW base"""
    param_groups = get_parameter_groups(model, config)
    
    return SAM(
        param_groups,
        base_optimizer=torch.optim.AdamW,
        rho=config.sam_rho,
        adaptive=config.sam_adaptive,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
        **kwargs
    )


@register_optimizer("sgd")
def create_sgd(
    model: nn.Module,
    config: RETFoundConfig,
    momentum: float = 0.9,
    nesterov: bool = True,
    **kwargs
) -> torch.optim.SGD:
    """Create SGD optimizer"""
    param_groups = get_parameter_groups(model, config)
    
    return torch.optim.SGD(
        param_groups,
        momentum=momentum,
        nesterov=nesterov,
        **kwargs
    )


def create_optimizer(
    model: nn.Module,
    config: RETFoundConfig,
    optimizer_name: Optional[str] = None,
    **kwargs
) -> Optimizer:
    """
    Create optimizer based on configuration
    
    Args:
        model: Model to optimize
        config: Configuration object
        optimizer_name: Specific optimizer to use
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Determine optimizer name
    if optimizer_name is None:
        if getattr(config, 'use_sam', False):
            optimizer_name = "sam_adamw"
        else:
            optimizer_name = getattr(config, 'optimizer', 'adamw')
    
    # Create optimizer
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Optimizer '{optimizer_name}' not found. "
            f"Available optimizers: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    
    optimizer_fn = OPTIMIZER_REGISTRY.get(optimizer_name)
    optimizer = optimizer_fn(model, config, **kwargs)
    
    logger.info(
        f"Created {optimizer_name} optimizer with "
        f"{len(optimizer.param_groups)} parameter groups"
    )
    
    return optimizer
