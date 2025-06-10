"""
Custom Samplers for RETFound - Dataset v6.1
==========================================

Implements advanced sampling strategies for handling class imbalance
and adaptive sampling during training, with specific support for
dataset v6.1 minority classes and critical conditions.
"""

import logging
from typing import Optional, List, Iterator, Union, Dict
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, WeightedRandomSampler, Subset

from ..core.registry import Registry
from ..core.constants import (
    CLASS_WEIGHTS_V61, CRITICAL_CONDITIONS,
    NUM_FUNDUS_CLASSES, NUM_OCT_CLASSES, NUM_TOTAL_CLASSES
)

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
    """Base class for all custom samplers with v6.1 support"""
    
    def __init__(self, dataset: Dataset, **kwargs):
        """
        Initialize sampler
        
        Args:
            dataset: Dataset to sample from
            **kwargs: Additional arguments
        """
        self.dataset = dataset
        
        # Handle Subset (for k-fold)
        if isinstance(dataset, Subset):
            self.base_dataset = dataset.dataset
            self.subset_indices = dataset.indices
            self.num_samples = len(dataset)
        else:
            self.base_dataset = dataset
            self.subset_indices = None
            self.num_samples = len(dataset)
        
        # Get labels
        self.labels = self._get_labels()
        
        # Get v6.1 specific info
        self.modality = self._get_modality()
        self.unified_classes = self._get_unified_classes()
        
        # Calculate class statistics
        self._calculate_class_stats()
        
        # Setup v6.1 specific weights
        self._setup_v61_weights()
    
    def _get_labels(self) -> List[int]:
        """Get labels from dataset"""
        if hasattr(self.dataset, 'targets'):
            labels = self.dataset.targets
        elif hasattr(self.dataset, 'get_labels'):
            labels = self.dataset.get_labels()
        elif hasattr(self.base_dataset, 'targets'):
            labels = self.base_dataset.targets
        else:
            raise ValueError("Dataset must have 'targets' attribute or 'get_labels' method")
        
        # Handle subset
        if self.subset_indices is not None:
            labels = [labels[i] for i in self.subset_indices]
        
        return labels
    
    def _get_modality(self) -> str:
        """Get dataset modality"""
        for dataset in [self.dataset, self.base_dataset]:
            if hasattr(dataset, 'modality'):
                return dataset.modality
        return 'both'  # default
    
    def _get_unified_classes(self) -> bool:
        """Check if using unified class system"""
        for dataset in [self.dataset, self.base_dataset]:
            if hasattr(dataset, 'unified_classes'):
                return dataset.unified_classes
        return True  # default for v6.1
    
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
        
        # Identify minority classes for v6.1
        self.minority_classes = self._identify_minority_classes()
        
        logger.info(
            f"Sampler initialized - Classes: {self.num_classes}, "
            f"Samples: {self.num_samples}, "
            f"Imbalance ratio: {self.imbalance_ratio:.1f}:1, "
            f"Modality: {self.modality}"
        )
    
    def _identify_minority_classes(self) -> Dict[int, float]:
        """Identify minority classes based on v6.1 configuration"""
        minority_classes = {}
        
        if self.unified_classes:
            # Unified class indices for minority classes
            minority_mapping = {
                22: 2.0,  # ERM (OCT class 4)
                25: 2.0,  # RVO_OCT (OCT class 7)
                27: 1.5,  # RAO_OCT (OCT class 9)
                12: 1.5,  # Myopia_Degenerative (Fundus class 12)
            }
            
            for class_idx, weight in minority_mapping.items():
                if class_idx in self.class_counts:
                    minority_classes[class_idx] = weight
        
        return minority_classes
    
    def _setup_v61_weights(self):
        """Setup dataset v6.1 specific weights"""
        self.v61_weights = {}
        
        # Base weight for all classes
        for class_idx in self.class_counts.keys():
            self.v61_weights[class_idx] = 1.0
        
        # Apply minority class weights
        for class_idx, weight in self.minority_classes.items():
            self.v61_weights[class_idx] = weight
        
        # Apply critical condition weights
        self._apply_critical_condition_weights()
    
    def _apply_critical_condition_weights(self):
        """Apply additional weights for critical conditions"""
        for condition, info in CRITICAL_CONDITIONS.items():
            if 'unified_idx' in info:
                for idx in info['unified_idx']:
                    if idx in self.v61_weights:
                        # Increase weight based on sensitivity requirement
                        sensitivity_factor = info['min_sensitivity']
                        self.v61_weights[idx] *= (1 + sensitivity_factor)
    
    def __len__(self) -> int:
        return self.num_samples


@register_sampler("balanced")
class BalancedSampler(BaseSampler):
    """
    Balanced sampler that ensures equal representation of all classes
    in each epoch by oversampling minority classes.
    Enhanced for v6.1 with minority class priority.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        samples_per_class: Optional[int] = None,
        minority_boost: float = 1.5,
        **kwargs
    ):
        """
        Initialize balanced sampler
        
        Args:
            dataset: Dataset to sample from
            samples_per_class: Number of samples per class per epoch
            minority_boost: Additional boost factor for minority classes
        """
        super().__init__(dataset, **kwargs)
        
        self.minority_boost = minority_boost
        
        # Determine samples per class
        if samples_per_class is None:
            # Use the size of the largest class
            self.base_samples_per_class = max(self.class_counts.values())
        else:
            self.base_samples_per_class = samples_per_class
        
        # Calculate samples per class with v6.1 adjustments
        self._calculate_samples_per_class()
        
        # Update total number of samples
        self.num_samples = sum(self.samples_per_class.values())
        
        logger.info(
            f"Balanced sampler v6.1: base {self.base_samples_per_class} samples/class, "
            f"total {self.num_samples} samples/epoch, "
            f"{len(self.minority_classes)} minority classes boosted"
        )
    
    def _calculate_samples_per_class(self):
        """Calculate samples per class with v6.1 minority boost"""
        self.samples_per_class = {}
        
        for class_idx in self.class_counts.keys():
            base_samples = self.base_samples_per_class
            
            # Apply minority boost if applicable
            if class_idx in self.minority_classes:
                boost_factor = self.minority_classes[class_idx] * self.minority_boost
                samples = int(base_samples * boost_factor)
            else:
                samples = base_samples
            
            self.samples_per_class[class_idx] = samples
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch"""
        indices = []
        
        # Sample from each class
        for class_label in sorted(self.class_indices.keys()):
            class_indices = self.class_indices[class_label]
            num_class_samples = len(class_indices)
            samples_needed = self.samples_per_class[class_label]
            
            # Sample with replacement if necessary
            if num_class_samples >= samples_needed:
                # Sample without replacement
                sampled_indices = np.random.choice(
                    class_indices,
                    size=samples_needed,
                    replace=False
                )
            else:
                # Sample with replacement (oversample)
                sampled_indices = np.random.choice(
                    class_indices,
                    size=samples_needed,
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
    with v6.1 specific adjustments
    """
    
    def __init__(
        self,
        dataset: Dataset,
        replacement: bool = True,
        num_samples: Optional[int] = None,
        use_v61_weights: bool = True,
        **kwargs
    ):
        """
        Initialize weighted sampler
        
        Args:
            dataset: Dataset to sample from
            replacement: Whether to sample with replacement
            num_samples: Number of samples to draw
            use_v61_weights: Whether to use v6.1 specific weights
        """
        super().__init__(dataset, **kwargs)
        
        self.replacement = replacement
        self.use_v61_weights = use_v61_weights
        if num_samples is not None:
            self.num_samples = num_samples
        
        # Calculate sample weights
        self._calculate_weights()
    
    def _calculate_weights(self):
        """Calculate weight for each sample with v6.1 adjustments"""
        # Calculate class weights (inverse frequency)
        total_samples = len(self.labels)
        class_weights = {}
        
        for class_label, count in self.class_counts.items():
            # Base weight (inverse frequency)
            weight = total_samples / (self.num_classes * count)
            
            # Apply v6.1 specific weights if enabled
            if self.use_v61_weights and class_label in self.v61_weights:
                weight *= self.v61_weights[class_label]
            
            class_weights[class_label] = weight
        
        # Assign weight to each sample
        self.weights = torch.zeros(total_samples)
        for idx, label in enumerate(self.labels):
            self.weights[idx] = class_weights[label]
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
        logger.info(
            f"Weighted sampler initialized with {'v6.1' if self.use_v61_weights else 'standard'} weights"
        )
    
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
    and class performance, with special focus on v6.1 critical conditions.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        initial_strategy: str = 'balanced',
        adaptation_rate: float = 0.1,
        min_samples_per_class: int = 10,
        critical_condition_boost: float = 2.0,
        **kwargs
    ):
        """
        Initialize adaptive sampler
        
        Args:
            dataset: Dataset to sample from
            initial_strategy: Initial sampling strategy
            adaptation_rate: Rate of adaptation (0-1)
            min_samples_per_class: Minimum samples per class
            critical_condition_boost: Boost for critical conditions
        """
        super().__init__(dataset, **kwargs)
        
        self.initial_strategy = initial_strategy
        self.adaptation_rate = adaptation_rate
        self.min_samples_per_class = min_samples_per_class
        self.critical_condition_boost = critical_condition_boost
        
        # Initialize class performance tracking
        self.class_performance = {
            label: 1.0 for label in self.class_indices.keys()
        }
        
        # Track critical condition performance
        self.critical_indices = self._get_critical_indices()
        
        # Initialize sampling probabilities
        self._update_sampling_probabilities()
        
        logger.info(
            f"Adaptive sampler v6.1 initialized with {initial_strategy} strategy, "
            f"monitoring {len(self.critical_indices)} critical classes"
        )
    
    def _get_critical_indices(self) -> List[int]:
        """Get indices of critical condition classes"""
        critical_indices = []
        
        if self.unified_classes:
            for condition, info in CRITICAL_CONDITIONS.items():
                if 'unified_idx' in info:
                    for idx in info['unified_idx']:
                        if idx in self.class_counts:
                            critical_indices.append(idx)
        
        return list(set(critical_indices))
    
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
            # Inverse frequency weighting with v6.1 weights
            total_samples = sum(self.class_counts.values())
            base_probs = {}
            for label, count in self.class_counts.items():
                weight = (total_samples / (self.num_classes * count))
                if label in self.v61_weights:
                    weight *= self.v61_weights[label]
                base_probs[label] = weight
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
            
            # Boost critical conditions if performing below threshold
            if label in self.critical_indices:
                for condition, info in CRITICAL_CONDITIONS.items():
                    if 'unified_idx' in info and label in info['unified_idx']:
                        if self.class_performance[label] < info['min_sensitivity']:
                            adapted_prob *= self.critical_condition_boost
                            break
            
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
            # Ensure minimum for minority classes
            if label in self.minority_classes:
                min_for_minority = int(self.min_samples_per_class * 1.5)
                samples = max(samples, min_for_minority)
            
            self.samples_per_class[label] = samples
    
    def update_performance(self, class_accuracies: Dict[int, float]):
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
        
        # Log critical condition status
        logger.info("Adaptive sampler - Critical conditions status:")
        for label in self.critical_indices:
            if label in self.class_performance:
                perf = self.class_performance[label]
                status = "✓" if perf >= 0.95 else "✗"
                logger.info(f"  Class {label}: {perf:.3f} {status}")
        
        # Update sampling probabilities
        self._update_sampling_probabilities()
    
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
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> Optional[Sampler]:
    """
    Create a sampler based on configuration with v6.1 support
    
    Args:
        dataset: Dataset to sample from
        sampler_type: Type of sampler to create
        balanced: Whether to use balanced sampling
        adaptive: Whether to use adaptive sampling
        class_weights: Pre-computed class weights (v6.1)
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
    
    # Pass class weights if available
    if class_weights is not None:
        kwargs['class_weights'] = class_weights
    
    # Create sampler
    sampler_class = SAMPLER_REGISTRY.get(sampler_type)
    sampler = sampler_class(dataset, **kwargs)
    
    logger.info(f"Created {sampler_type} sampler for dataset v6.1")
    
    return sampler


def create_weighted_random_sampler(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    use_v61_weights: bool = True
) -> WeightedRandomSampler:
    """
    Create PyTorch's WeightedRandomSampler for dataset v6.1
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to draw
        use_v61_weights: Whether to apply v6.1 specific weights
        
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
        
        # Apply v6.1 minority class weights if enabled
        if use_v61_weights:
            # Map to v6.1 weights
            v61_weight = 1.0
            if class_label == 22:  # ERM
                v61_weight = 2.0
            elif class_label == 25:  # RVO_OCT
                v61_weight = 2.0
            elif class_label == 27:  # RAO_OCT
                v61_weight = 1.5
            elif class_label == 12:  # Myopia_Degenerative
                v61_weight = 1.5
            
            weight *= v61_weight
        
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
        f"Created WeightedRandomSampler for v6.1 with {num_classes} classes"
    )
    
    return sampler