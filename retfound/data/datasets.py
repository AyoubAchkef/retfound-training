"""
Dataset Classes for RETFound
===========================

Implements dataset classes for loading retinal images with advanced features
including caching, pathology-aware augmentation, and support for dataset v6.1.
"""

import os
import sys
import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..core.registry import Registry
from ..core.exceptions import DatasetNotFoundError, DataCorruptedError
from ..core.constants import (
    FUNDUS_CLASS_NAMES, OCT_CLASS_NAMES, UNIFIED_CLASS_NAMES,
    FUNDUS_CLASS_TO_IDX, OCT_CLASS_TO_IDX, UNIFIED_CLASS_TO_IDX,
    NUM_FUNDUS_CLASSES, NUM_OCT_CLASSES, NUM_TOTAL_CLASSES,
    IMAGE_EXTENSIONS, DATASET_V61_STATS
)

logger = logging.getLogger(__name__)

# Dataset registry
DATASET_REGISTRY = Registry("datasets")


def register_dataset(name: str):
    """Decorator to register a dataset"""
    def decorator(cls):
        DATASET_REGISTRY.register(name, cls)
        return cls
    return decorator


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets"""
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Initialize data
        self.data = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        
        # Load dataset
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """Load dataset - must be implemented by subclasses"""
        pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item by index"""
        img_path = self.data[index]
        target = self.targets[index]
        
        # Load image
        image = self._load_image(img_path)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image with fallback methods"""
        try:
            if CV2_AVAILABLE:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("CV2 failed to load image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(path).convert('RGB')
                image = np.array(image)
        except Exception as e:
            raise DataCorruptedError(f"Failed to load image {path}: {e}")
        
        return image
    
    def get_labels(self) -> List[int]:
        """Get all labels"""
        return self.targets
    
    def get_classes(self) -> List[str]:
        """Get class names"""
        return self.classes


@register_dataset("retfound_v61")
@register_dataset("caasi_v61")
class CAASIDatasetV61(BaseDataset):
    """
    CAASI Dataset v6.1 - Supports both Fundus and OCT images
    
    Expected structure:
    root/
    ├── fundus/
    │   ├── train/
    │   │   ├── 00_Normal_Fundus/
    │   │   ├── 01_DR_Mild/
    │   │   └── ... (18 classes)
    │   ├── val/
    │   └── test/
    └── oct/
        ├── train/
        │   ├── 00_Normal_OCT/
        │   ├── 01_DME/
        │   └── ... (10 classes)
        ├── val/
        └── test/
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        modality: str = 'both',  # 'fundus', 'oct', or 'both'
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        pathology_augmentation: Optional[Any] = None,
        unified_classes: bool = True,  # Use unified 28-class system
        return_metadata: bool = False,
        **kwargs
    ):
        """
        Initialize CAASI Dataset v6.1
        
        Args:
            root: Root directory containing 'fundus' and 'oct' folders
            split: Dataset split ('train', 'val', 'test')
            modality: Which images to load ('fundus', 'oct', or 'both')
            transform: Image transform
            target_transform: Target transform
            use_cache: Whether to use caching
            cache_dir: Cache directory
            pathology_augmentation: Pathology-specific augmentation
            unified_classes: Use unified 28-class system
            return_metadata: Return additional metadata with each sample
        """
        self.modality = modality
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pathology_augmentation = pathology_augmentation
        self.unified_classes = unified_classes
        self.return_metadata = return_metadata
        
        # Additional data storage
        self.modalities = []  # Store modality for each sample
        self.num_classes = 0
        
        # Set up cache
        if self.use_cache and self.cache_dir:
            self.cache_dir = self.cache_dir / f"{modality}_{split}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(root, split, transform, target_transform, **kwargs)
        
        # Calculate class weights for balanced sampling
        self._calculate_class_weights()
    
    def _load_data(self):
        """Load dataset following v6.1 structure"""
        loaded_fundus = 0
        loaded_oct = 0
        
        # Load fundus images
        if self.modality in ['fundus', 'both']:
            fundus_dir = self.root / 'fundus' / self.split
            if fundus_dir.exists():
                loaded_fundus = self._load_modality(
                    fundus_dir, 'fundus', FUNDUS_CLASS_NAMES, FUNDUS_CLASS_TO_IDX
                )
            else:
                logger.warning(f"Fundus directory not found: {fundus_dir}")
        
        # Load OCT images
        if self.modality in ['oct', 'both']:
            oct_dir = self.root / 'oct' / self.split
            if oct_dir.exists():
                loaded_oct = self._load_modality(
                    oct_dir, 'oct', OCT_CLASS_NAMES, OCT_CLASS_TO_IDX
                )
            else:
                logger.warning(f"OCT directory not found: {oct_dir}")
        
        # Validate dataset
        if not self.data:
            raise DatasetNotFoundError(
                f"No images found for modality '{self.modality}' in split '{self.split}'"
            )
        
        # Set up classes based on mode
        self._setup_classes()
        
        # Log dataset info
        logger.info(f"Loaded {loaded_fundus} fundus and {loaded_oct} OCT images")
        logger.info(f"Total samples: {len(self.data)}, Classes: {self.num_classes}")
        self._log_dataset_info()
    
    def _load_modality(
        self, 
        base_dir: Path, 
        modality: str, 
        class_names: List[str],
        class_to_idx: Dict[str, int]
    ) -> int:
        """Load images from a single modality"""
        count = 0
        
        for class_name in class_names:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all valid images
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            
            for img_path in image_files:
                # Original class index within modality
                modality_class_idx = class_to_idx[class_name]
                
                # Calculate unified class index if needed
                if self.unified_classes:
                    if modality == 'fundus':
                        unified_idx = modality_class_idx
                    else:  # oct
                        unified_idx = NUM_FUNDUS_CLASSES + modality_class_idx
                else:
                    unified_idx = modality_class_idx
                
                self.data.append(str(img_path))
                self.targets.append(unified_idx)
                self.modalities.append(modality)
                count += 1
        
        return count
    
    def _setup_classes(self):
        """Set up class names and mappings based on configuration"""
        if self.unified_classes:
            self.classes = UNIFIED_CLASS_NAMES
            self.class_to_idx = UNIFIED_CLASS_TO_IDX
            self.num_classes = NUM_TOTAL_CLASSES
        else:
            if self.modality == 'fundus':
                self.classes = FUNDUS_CLASS_NAMES
                self.class_to_idx = FUNDUS_CLASS_TO_IDX
                self.num_classes = NUM_FUNDUS_CLASSES
            elif self.modality == 'oct':
                self.classes = OCT_CLASS_NAMES
                self.class_to_idx = OCT_CLASS_TO_IDX
                self.num_classes = NUM_OCT_CLASSES
            else:  # both but not unified
                # Combine class names
                self.classes = FUNDUS_CLASS_NAMES + OCT_CLASS_NAMES
                self.num_classes = NUM_FUNDUS_CLASSES + NUM_OCT_CLASSES
                # Create new mapping
                self.class_to_idx = {
                    name: idx for idx, name in enumerate(self.classes)
                }
        
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
    
    def _calculate_class_weights(self):
        """Calculate class weights for balanced sampling"""
        class_counts = defaultdict(int)
        for label in self.targets:
            class_counts[label] += 1
        
        # Calculate weights
        total_samples = len(self.targets)
        num_active_classes = len(class_counts)
        
        self.class_weights = torch.zeros(self.num_classes)
        for class_idx, count in class_counts.items():
            weight = total_samples / (num_active_classes * count)
            self.class_weights[class_idx] = weight
        
        # Normalize weights
        self.class_weights = self.class_weights / self.class_weights.mean()
    
    def _log_dataset_info(self):
        """Log dataset information"""
        class_counts = defaultdict(int)
        modality_counts = defaultdict(int)
        
        for i, label in enumerate(self.targets):
            class_counts[label] += 1
            modality_counts[self.modalities[i]] += 1
        
        logger.info(f"\nDataset Summary for {self.split}:")
        logger.info(f"   Total images: {len(self.data):,}")
        logger.info(f"   Modality: {self.modality}")
        logger.info(f"   Number of classes: {self.num_classes}")
        
        if self.modality == 'both':
            logger.info(f"   Fundus images: {modality_counts['fundus']:,}")
            logger.info(f"   OCT images: {modality_counts['oct']:,}")
        
        logger.info(f"   Class distribution:")
        
        for label in sorted(class_counts.keys()):
            count = class_counts[label]
            class_name = self.classes[label] if label < len(self.classes) else f"Class_{label}"
            percentage = (count / len(self.data)) * 100
            logger.info(
                f"      {class_name}: {count:,} ({percentage:.1f}%)"
            )
    
    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, Dict]]:
        """Get item with caching and metadata support"""
        img_path = self.data[index]
        target = self.targets[index]
        modality = self.modalities[index]
        
        # Try cache first
        if self.use_cache and self.cache_dir:
            image = self._load_from_cache(img_path)
        else:
            image = self._load_image(img_path)
        
        # Apply pathology-specific augmentation
        if self.pathology_augmentation and self.split == 'train':
            class_name = self.classes[target] if target < len(self.classes) else ''
            pathology_aug = self.pathology_augmentation.get_augmentation(class_name)
            if pathology_aug:
                image = pathology_aug(image=image)['image']
        
        # Apply transforms
        if self.transform is not None:
            # Handle different transform types
            if hasattr(self.transform, 'transform'):  # Albumentations
                augmented = self.transform(image=image)
                image = augmented['image']
            else:  # torchvision
                image = Image.fromarray(image)
                image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Return based on configuration
        if self.return_metadata:
            metadata = {
                'path': img_path,
                'modality': modality,
                'class_name': self.classes[target] if target < len(self.classes) else str(target)
            }
            return image, target, metadata
        else:
            return image, target
    
    def _load_from_cache(self, img_path: str) -> np.ndarray:
        """Load image from cache or create cache entry"""
        cache_key = hashlib.md5(img_path.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # Cache corrupted, reload
                pass
        
        # Load and cache image
        image = self._load_image(img_path)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(image, f)
        except Exception:
            # Cache write failed, continue without caching
            pass
        
        return image
    
    def get_class_weights(self) -> torch.Tensor:
        """Get calculated class weights for balanced training"""
        return self.class_weights
    
    def get_imbalance_ratio(self) -> float:
        """Get imbalance ratio (max/min class frequency)"""
        class_counts = np.bincount(self.targets)
        class_counts = class_counts[class_counts > 0]  # Remove empty classes
        
        if len(class_counts) < 2:
            return 1.0
        
        return float(class_counts.max()) / float(class_counts.min())
    
    def get_subset_by_modality(self, modality: str) -> 'CAASIDatasetV61':
        """Get a subset of the dataset for a specific modality"""
        if modality not in ['fundus', 'oct']:
            raise ValueError(f"Invalid modality: {modality}")
        
        # Filter indices
        indices = [
            i for i, m in enumerate(self.modalities) if m == modality
        ]
        
        # Create new dataset with filtered data
        subset = CAASIDatasetV61(
            root=self.root,
            split=self.split,
            modality=modality,
            transform=self.transform,
            target_transform=self.target_transform,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir.parent if self.cache_dir else None,
            pathology_augmentation=self.pathology_augmentation,
            unified_classes=self.unified_classes,
            return_metadata=self.return_metadata
        )
        
        # Override data with filtered subset
        subset.data = [self.data[i] for i in indices]
        subset.targets = [self.targets[i] for i in indices]
        subset.modalities = [self.modalities[i] for i in indices]
        
        return subset


# Register legacy names for backward compatibility
@register_dataset("retfound")
@register_dataset("caasi")
class RETFoundDataset(CAASIDatasetV61):
    """Legacy dataset class for backward compatibility"""
    
    def __init__(self, *args, **kwargs):
        logger.warning(
            "RETFoundDataset is deprecated. Use CAASIDatasetV61 instead."
        )
        super().__init__(*args, **kwargs)


def create_dataset(
    dataset_name: str,
    root: Union[str, Path],
    split: str = 'train',
    **kwargs
) -> BaseDataset:
    """
    Create a dataset from registry
    
    Args:
        dataset_name: Name of the dataset
        root: Root directory
        split: Dataset split
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    if dataset_name not in DATASET_REGISTRY:
        raise DatasetNotFoundError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    return dataset_class(root=root, split=split, **kwargs)


def list_datasets() -> List[str]:
    """List all available datasets"""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset"""
    if dataset_name not in DATASET_REGISTRY:
        raise DatasetNotFoundError(f"Dataset '{dataset_name}' not found")
    
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    
    return {
        'name': dataset_name,
        'class': dataset_class.__name__,
        'description': dataset_class.__doc__ or "No description available"
    }


def get_dataset_stats(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """Get statistics about the dataset v6.1"""
    dataset_path = Path(dataset_path)
    stats = {
        'fundus': {'train': 0, 'val': 0, 'test': 0, 'total': 0},
        'oct': {'train': 0, 'val': 0, 'test': 0, 'total': 0},
        'total': {'train': 0, 'val': 0, 'test': 0, 'total': 0},
        'classes': {'fundus': {}, 'oct': {}}
    }
    
    for modality in ['fundus', 'oct']:
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / modality / split
            if not split_dir.exists():
                continue
            
            # Count images per class
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                count = len([
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ])
                
                stats[modality][split] += count
                stats['total'][split] += count
                
                if class_dir.name not in stats['classes'][modality]:
                    stats['classes'][modality][class_dir.name] = {
                        'train': 0, 'val': 0, 'test': 0, 'total': 0
                    }
                stats['classes'][modality][class_dir.name][split] = count
    
    # Calculate totals
    for modality in ['fundus', 'oct', 'total']:
        stats[modality]['total'] = sum(
            stats[modality][split] for split in ['train', 'val', 'test']
        )
    
    return stats