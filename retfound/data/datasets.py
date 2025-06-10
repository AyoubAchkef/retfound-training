"""
Dataset Classes for RETFound
===========================

Implements dataset classes for loading retinal images with advanced features
including caching, pathology-aware augmentation, and automatic class detection.
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


@register_dataset("retfound")
@register_dataset("caasi")
class RETFoundDataset(BaseDataset):
    """
    RETFound dataset with CAASI structure support
    
    Expected structure:
    root/
    ├── train/
    │   ├── 0_normal/
    │   ├── 1_dr_mild/
    │   └── ...
    ├── val/
    └── test/
    """
    
    # CAASI-specific class name mapping
    CAASI_CLASS_MAPPING = {
        # Diabetic Retinopathy stages
        'normal': 0, '0_normal': 0, 'normal_fundus': 0,
        'mild': 1, '1_mild': 1, 'dr_mild': 1, '1_dr_mild': 1,
        'moderate': 2, '2_moderate': 2, 'dr_moderate': 2, '2_dr_moderate': 2,
        'severe': 3, '3_severe': 3, 'dr_severe': 3, '3_dr_severe': 3,
        'proliferative': 4, '4_proliferative': 4, 'dr_proliferative': 4, '4_dr_proliferative': 4,
        
        # OCT pathologies
        'normal_oct': 5, '5_normal_oct': 5,
        'amd': 6, '6_amd': 6, 'age_related_macular_degeneration': 6,
        'dme': 7, '7_dme': 7, 'diabetic_macular_edema': 7, 'dme_fundus': 7,
        'erm': 8, '8_erm': 8, 'epiretinal_membrane': 8,
        'rao': 9, '9_rao': 9, 'retinal_artery_occlusion': 9,
        'rvo': 10, '10_rvo': 10, 'retinal_vein_occlusion': 10,
        'vid': 11, '11_vid': 11, 'vitreomacular_interface_disease': 11,
        'cnv': 12, '12_cnv': 12, 'choroidal_neovascularization': 12,
        'dme_oct': 13, '13_dme_oct': 13,
        'drusen': 14, '14_drusen': 14,
        
        # Glaucoma
        'glaucoma_normal': 15, '15_glaucoma_normal': 15,
        'glaucoma_positive': 16, '16_glaucoma_positive': 16,
        'glaucoma_suspect': 17, '17_glaucoma_suspect': 17,
        
        # Additional conditions
        'myopia': 18, '18_myopia': 18,
        'cataract': 19, '19_cataract': 19,
        'retinal_detachment': 20, '20_retinal_detachment': 20,
        'other': 21, '21_other': 21
    }
    
    # Proper class names for display
    CLASS_NAMES = {
        0: 'Normal_Fundus',
        1: 'DR_Mild',
        2: 'DR_Moderate', 
        3: 'DR_Severe',
        4: 'DR_Proliferative',
        5: 'Normal_OCT',
        6: 'AMD',
        7: 'DME_Fundus',
        8: 'ERM',
        9: 'RAO',
        10: 'RVO',
        11: 'VID',
        12: 'CNV',
        13: 'DME_OCT',
        14: 'DRUSEN',
        15: 'Glaucoma_Normal',
        16: 'Glaucoma_Positive',
        17: 'Glaucoma_Suspect',
        18: 'Myopia',
        19: 'Cataract',
        20: 'Retinal_Detachment',
        21: 'Other'
    }
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        pathology_augmentation: Optional[Any] = None,
        image_extensions: Optional[set] = None,
        **kwargs
    ):
        """
        Initialize RETFound dataset
        
        Args:
            root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transform
            target_transform: Target transform
            use_cache: Whether to use caching
            cache_dir: Cache directory
            pathology_augmentation: Pathology-specific augmentation
            image_extensions: Valid image extensions
        """
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pathology_augmentation = pathology_augmentation
        self.image_extensions = image_extensions or {
            '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'
        }
        
        # Set up cache
        if self.use_cache and self.cache_dir:
            self.cache_dir = self.cache_dir / split
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(root, split, transform, target_transform, **kwargs)
    
    def _load_data(self):
        """Load dataset following CAASI structure"""
        split_dir = self.root / self.split
        
        if not split_dir.exists():
            raise DatasetNotFoundError(
                f"Split directory not found: {split_dir}"
            )
        
        # Check for class subfolders
        subfolders = sorted([
            f for f in split_dir.iterdir()
            if f.is_dir() and not f.name.startswith('.')
        ])
        
        if not subfolders:
            # Try flat structure with CSV labels
            self._load_flat_structure(split_dir)
        else:
            # Load hierarchical structure
            self._load_hierarchical_structure(split_dir, subfolders)
        
        # Validate dataset
        if not self.data:
            raise DatasetNotFoundError(
                f"No images found in {split_dir}"
            )
        
        # Set up classes
        self._setup_classes()
        
        # Log dataset info
        self._log_dataset_info()
    
    def _load_hierarchical_structure(self, split_dir: Path, subfolders: List[Path]):
        """Load dataset with class folders"""
        class_counts = defaultdict(int)
        
        logger.info(f"Loading {self.split} with {len(subfolders)} class folders")
        
        for class_folder in subfolders:
            class_name = class_folder.name.lower()
            
            # Map class name to label
            label = self._get_label_from_name(class_name)
            if label is None:
                logger.warning(f"Unknown class: {class_name}, skipping...")
                continue
            
            # Load images from class folder
            img_count = 0
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    self.data.append(str(img_path))
                    self.targets.append(label)
                    img_count += 1
            
            class_counts[label] += img_count
            if img_count > 0:
                logger.info(
                    f"   {class_folder.name}: {img_count} images → label {label}"
                )
    
    def _load_flat_structure(self, split_dir: Path):
        """Load dataset with flat structure and CSV labels"""
        # Look for labels CSV
        labels_file = split_dir / 'labels.csv'
        if labels_file.exists():
            self._load_from_csv(split_dir, labels_file)
        else:
            raise DatasetNotFoundError(
                f"No class folders or labels.csv found in {split_dir}"
            )
    
    def _load_from_csv(self, split_dir: Path, labels_file: Path):
        """Load dataset from CSV file"""
        import csv
        
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = split_dir / row['filename']
                if img_path.exists():
                    self.data.append(str(img_path))
                    self.targets.append(int(row['label']))
    
    def _get_label_from_name(self, class_name: str) -> Optional[int]:
        """Get label from class name"""
        # Try direct mapping
        if class_name in self.CAASI_CLASS_MAPPING:
            return self.CAASI_CLASS_MAPPING[class_name]
        
        # Try to extract numeric prefix
        if '_' in class_name:
            try:
                return int(class_name.split('_')[0])
            except ValueError:
                pass
        
        return None
    
    def _setup_classes(self):
        """Set up class names and mappings"""
        unique_labels = sorted(set(self.targets))
        max_label = max(unique_labels) if unique_labels else 0
        
        # Create class names
        self.classes = []
        for i in range(max_label + 1):
            if i in self.CLASS_NAMES:
                self.classes.append(self.CLASS_NAMES[i])
            else:
                self.classes.append(f'Class_{i}')
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
    
    def _log_dataset_info(self):
        """Log dataset information"""
        class_counts = defaultdict(int)
        for label in self.targets:
            class_counts[label] += 1
        
        logger.info(f"\nDataset Summary for {self.split}:")
        logger.info(f"   Total images: {len(self.data):,}")
        logger.info(f"   Number of classes: {len(set(self.targets))}")
        logger.info(f"   Class distribution:")
        
        for label in sorted(class_counts.keys()):
            count = class_counts[label]
            class_name = self.classes[label] if label < len(self.classes) else f"Class_{label}"
            percentage = (count / len(self.data)) * 100
            logger.info(
                f"      {class_name} (label {label}): "
                f"{count:,} images ({percentage:.1f}%)"
            )
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item with caching support"""
        img_path = self.data[index]
        target = self.targets[index]
        
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
        """Calculate class weights for imbalanced datasets"""
        class_counts = np.bincount(self.targets)
        total_samples = len(self.targets)
        num_classes = len(class_counts)
        
        # Calculate weights
        weights = total_samples / (num_classes * class_counts)
        
        # Normalize
        weights = weights / weights.mean()
        
        return torch.FloatTensor(weights)
    
    def get_imbalance_ratio(self) -> float:
        """Get imbalance ratio (max/min class frequency)"""
        class_counts = np.bincount(self.targets)
        class_counts = class_counts[class_counts > 0]  # Remove empty classes
        
        if len(class_counts) < 2:
            return 1.0
        
        return float(class_counts.max()) / float(class_counts.min())


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
