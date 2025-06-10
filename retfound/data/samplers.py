"""
Custom Samplers for RETFound
============================

Implements advanced sampling strategies for handling class imbalance
and adaptive sampling during training.
"""

import logging
from typing import Optional, List, Iterator, Union
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, WeightedRandomSampler

from ..core.registry import Registry

logger = logging.getLogger(__name__)

# Sampler registry
SAMPLER_REGISTRY = Registry("samplers")


def register_sampler(name: str):
    """Decorator to register a sampler"""
    def decorator(cls):
        SAMPLER_REGISTRY.register(name, cls)
        return cls
    return decorator


class BaseSampler(Sampler):
    """Base class for all custom samplers"""
    
    def __init__(self, dataset: Dataset, **kwargs):
        """
        Initialize sampler
        
        Args:
            dataset: Dataset to sample from
            **kwargs: Additional arguments
        """
        self.dataset = dataset
        self.num_samples = len(dataset)
        
        # Get labels
        if hasattr(dataset, 'targets'):
            self.labels = dataset.targets
        elif hasattr(dataset, 'get_labels'):
            self.labels = dataset.get_labels()
        else:
            raise ValueError("Dataset must have 'targets' attribute or 'get_labels' method")
        
        # Calculate class statistics
        self._calculate_class_stats()
    
    def _calculate_class_stats(self):
        """Calculate class distribution statistics"""
        self.class_counts = Counter(self.labels)
        self.num_classes = len(self.class_counts)
        self.class_indices = {}
        
        # Group indices by class
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate imbalance ratio
        counts = list(self.class_counts.values())
        self.imbalance_ratio = max(counts) / min(counts) if counts else 1.0
        
        logger.info(
            f"Sampler initialized - Classes: {self.num_classes}, "
            f"Samples: {self.num_samples}, "
            f"Imbalance ratio: {self.imbalance_ratio:.1f}:1"
        )
    
    def __len__(self) -> int:
        return self.num_samples


@register_sampler("balanced")
class BalancedSampler(BaseSampler):
    """
    Balanced sampler that ensures equal representation of all classes
    in each epoch by oversampling minority classes.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        samples_per_class: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize balanced sampler
        
        Args:
            dataset: Dataset to sample from
            samples_per_class: Number of samples per class per epoch
                              If None, uses the size of the largest class
        """
        super().__init__(dataset, **kwargs)
        
        # Determine samples per class
        if samples_per_class is None:
            # Use the size of the largest class
            self.samples_per_class = max(self.class_counts.values())
        else:
            self.samples_per_class = samples_per_class
        
        # Update total number of samples
        self.num_samples = self.samples_per_class * self.num_classes
        
        logger.info(
            f"Balanced sampler: {self.samples_per_class} samples per class, "
            f"total {self.num_samples} samples per epoch"
        )
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch"""
        indices = []
        
        # Sample from each class
        for class_label in sorted(self.class_indices.keys()):
            class_indices = self.class_indices[class_label]
            num_class_samples = len(class_indices)
            
            # Sample with replacement if necessary
            if num_class_samples >= self.samples_per_class:
                # Sample without replacement
                sampled_indices = np.random.choice(
                    class_indices,
                    size=self.samples_per_class,
                    replace=False
                )
            else:
                # Sample with replacement (oversample)
                sampled_indices = np.random.choice(
                    class_indices,
                    size=self.samples_per_class,
                    replace=True
                )
            
            indices.extend(sampled_indices)
        
        # Shuffle all indices
        np.random.shuffle(indices)
        
        return iter(indices)


@register_sampler("weighted")
class WeightedSampler(BaseSampler):
    """
    Weighted random sampler that samples based on inverse class frequency
    """
    
    def __init__(
        self,
        dataset: Dataset,
        replacement: bool = True,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize weighted sampler
        
        Args:
            dataset: Dataset to sample from
            replacement: Whether to sample with replacement
            num_samples: Number of samples to draw
        """
        super().__init__(dataset, **kwargs)
        
        self.replacement = replacement
        if num_samples is not None:
            self.num_samples = num_samples
        
        # Calculate sample weights
        self._calculate_weights()
    
    def _calculate_weights(self):
        """Calculate weight for each sample"""
        # Calculate class weights (inverse frequency)
        total_samples = len(self.labels)
        class_weights = {}
        
        for class_label, count in self.class_counts.items():
            weight = total_samples / (self.num_classes * count)
            class_weights[class_label] = weight
        
        # Assign weight to each sample
        self.weights = torch.zeros(total_samples)
        for idx, label in enumerate(self.labels):
            self.weights[idx] = class_weights[label]
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
        logger.info("Weighted sampler initialized with inverse class frequency weights")
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch"""
        # Use PyTorch's multinomial sampling
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement
        ).tolist()
        
        return iter(indices)


@register_sampler("adaptive")
class AdaptiveSampler(BaseSampler):
    """
    Adaptive sampler that adjusts sampling strategy based on training progress
    and class performance.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        initial_strategy: str = 'balanced',
        adaptation_rate: float = 0.1,
        min_samples_per_class: int = 10,
        **kwargs
    ):
        """
        Initialize adaptive sampler
        
        Args:
            dataset: Dataset to sample from
            initial_strategy: Initial sampling strategy ('balanced', 'weighted', 'uniform')
            adaptation_rate: Rate of adaptation (0-1)
            min_samples_per_class: Minimum samples per class
        """
        super().__init__(dataset, **kwargs)
        
        self.initial_strategy = initial_strategy
        self.adaptation_rate = adaptation_rate
        self.min_samples_per_class = min_samples_per_class
        
        # Initialize class performance tracking
        self.class_performance = {
            label: 1.0 for label in self.class_indices.keys()
        }
        
        # Initialize sampling probabilities
        self._update_sampling_probabilities()
        
        logger.info(
            f"Adaptive sampler initialized with {initial_strategy} strategy"
        )
    
    def _update_sampling_probabilities(self):
        """Update sampling probabilities based on class performance"""
        # Calculate base probabilities based on strategy
        if self.initial_strategy == 'balanced':
            # Equal probability for all classes
            base_probs = {
                label: 1.0 / self.num_classes
                for label in self.class_indices.keys()
            }
        elif self.initial_strategy == 'weighted':
            # Inverse frequency weighting
            total_samples = sum(self.class_counts.values())
            base_probs = {
                label: (total_samples / (self.num_classes * count))
                for label, count in self.class_counts.items()
            }
            # Normalize
            prob_sum = sum(base_probs.values())
            base_probs = {k: v/prob_sum for k, v in base_probs.items()}
        else:  # uniform
            # Proportional to class size
            total_samples = sum(self.class_counts.values())
            base_probs = {
                label: count / total_samples
                for label, count in self.class_counts.items()
            }
        
        # Adapt based on performance
        self.class_probabilities = {}
        for label in self.class_indices.keys():
            # Blend base probability with performance-based adjustment
            performance_factor = 1.0 / (self.class_performance[label] + 1e-6)
            adapted_prob = (
                (1 - self.adaptation_rate) * base_probs[label] +
                self.adaptation_rate * performance_factor
            )
            self.class_probabilities[label] = adapted_prob
        
        # Normalize probabilities
        prob_sum = sum(self.class_probabilities.values())
        self.class_probabilities = {
            k: v/prob_sum for k, v in self.class_probabilities.items()
        }
        
        # Calculate samples per class
        self.samples_per_class = {}
        for label, prob in self.class_probabilities.items():
            samples = max(
                int(prob * self.num_samples),
                self.min_samples_per_class
            )
            self.samples_per_class[label] = samples
    
    def update_performance(self, class_accuracies: dict):
        """
        Update class performance metrics
        
        Args:
            class_accuracies: Dictionary mapping class labels to accuracies
        """
        for label, accuracy in class_accuracies.items():
            if label in self.class_performance:
                # Exponential moving average
                self.class_performance[label] = (
                    0.9 * self.class_performance[label] + 0.1 * accuracy
                )
        
        # Update sampling probabilities
        self._update_sampling_probabilities()
        
        logger.info("Adaptive sampler updated with new class performance metrics")
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch"""
        indices = []
        
        # Sample from each class based on adaptive probabilities
        for class_label, num_samples in self.samples_per_class.items():
            class_indices = self.class_indices[class_label]
            num_class_samples = len(class_indices)
            
            # Sample with or without replacement as needed
            replace = num_samples > num_class_samples
            sampled_indices = np.random.choice(
                class_indices,
                size=num_samples,
                replace=replace
            )
            
            indices.extend(sampled_indices)
        
        # Shuffle all indices
        np.random.shuffle(indices)
        
        # Trim to exact number of samples if needed
        if len(indices) > self.num_samples:
            indices = indices[:self.num_samples]
        
        return iter(indices)


def create_sampler(
    dataset: Dataset,
    sampler_type: Optional[str] = None,
    balanced: bool = False,
    adaptive: bool = False,
    **kwargs
) -> Optional[Sampler]:
    """
    Create a sampler based on configuration
    
    Args:
        dataset: Dataset to sample from
        sampler_type: Type of sampler to create
        balanced: Whether to use balanced sampling
        adaptive: Whether to use adaptive sampling
        **kwargs: Additional arguments for sampler
        
    Returns:
        Sampler instance or None
    """
    # Determine sampler type
    if sampler_type:
        if sampler_type not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown sampler type: {sampler_type}. "
                f"Available: {list(SAMPLER_REGISTRY.keys())}"
            )
    elif adaptive:
        sampler_type = 'adaptive'
    elif balanced:
        sampler_type = 'balanced'
    else:
        return None
    
    # Create sampler
    sampler_class = SAMPLER_REGISTRY.get(sampler_type)
    sampler = sampler_class(dataset, **kwargs)
    
    logger.info(f"Created {sampler_type} sampler")
    
    return sampler


def create_weighted_random_sampler(
    dataset: Dataset,
    num_samples: Optional[int] = None
) -> WeightedRandomSampler:
    """
    Create PyTorch's WeightedRandomSampler for a dataset
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to draw
        
    Returns:
        WeightedRandomSampler instance
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'get_labels'):
        labels = dataset.get_labels()
    else:
        raise ValueError("Dataset must have 'targets' attribute or 'get_labels' method")
    
    # Calculate class weights
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    
    # Calculate weight for each class
    class_weights = {}
    for class_label, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_label] = weight
    
    # Create sample weights
    weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples or total_samples,
        replacement=True
    )
    
    logger.info(
        f"Created WeightedRandomSampler with {num_classes} classes"
    )
    
    return sampler
