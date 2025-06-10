"""
DataModule for RETFound - Dataset v6.1
=====================================

Lightning-style DataModule for managing data loading, transforms,
and dataset splits for Dataset v6.1 (28 classes).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np

from ..core.config import RETFoundConfig
from ..core.constants import (
    NUM_TOTAL_CLASSES, UNIFIED_CLASS_NAMES,
    FUNDUS_CLASS_NAMES, OCT_CLASS_NAMES,
    NUM_FUNDUS_CLASSES, NUM_OCT_CLASSES,
    CLASS_WEIGHTS_V61, DATASET_V61_STATS
)
from .datasets import create_dataset, CAASIDatasetV61
from .transforms import (
    create_train_transforms,
    create_val_transforms,
    create_test_transforms,
    PathologyAugmentation
)
from .samplers import create_sampler

logger = logging.getLogger(__name__)


class RETFoundDataModule:
    """
    DataModule for RETFound training on Dataset v6.1
    
    Handles:
    - Dataset creation and splitting
    - Transform application
    - DataLoader creation
    - Sampling strategies
    - Multi-modality support (Fundus/OCT/Both)
    - K-fold cross-validation
    """
    
    def __init__(
        self,
        config: RETFoundConfig,
        dataset_name: str = 'caasi_v61',
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize DataModule for v6.1
        
        Args:
            config: Configuration object
            dataset_name: Name of dataset to use (default: 'caasi_v61')
            train_transforms: Custom train transforms
            val_transforms: Custom validation transforms
            test_transforms: Custom test transforms
            **kwargs: Additional arguments passed to dataset
        """
        self.config = config
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        
        # Extract dataset configuration
        self.modality = self._get_config_value('modality', 'both')
        self.unified_classes = self._get_config_value('unified_classes', True)
        self.use_cache = self._get_config_value('use_cache', True)
        self.balance_dataset = self._get_config_value('balance_dataset', True)
        self.adaptive_sampling = self._get_config_value('adaptive_sampling', True)
        self.return_metadata = self._get_config_value('return_metadata', False)
        
        # Datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # DataLoaders
        self.train_dataloader_obj: Optional[DataLoader] = None
        self.val_dataloader_obj: Optional[DataLoader] = None
        self.test_dataloader_obj: Optional[DataLoader] = None
        
        # Transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        
        # Pathology augmentation
        self.pathology_aug = None
        if self._get_config_value('use_pathology_augmentation', True):
            input_size = self._get_config_value('input_size', 224)
            self.pathology_aug = PathologyAugmentation(input_size)
        
        # Setup transforms if not provided
        self._setup_transforms()
        
        # Dataset info
        self.num_classes: Optional[int] = None
        self.class_names: Optional[List[str]] = None
        self.class_weights: Optional[torch.Tensor] = None
        
        # K-fold support
        self.n_folds: Optional[int] = None
        self.current_fold: Optional[int] = None
        self.fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    
    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get value from nested or flat config"""
        # Try nested config first
        if hasattr(self.config, 'data') and hasattr(self.config.data, key):
            return getattr(self.config.data, key)
        elif hasattr(self.config, 'training') and hasattr(self.config.training, key):
            return getattr(self.config.training, key)
        # Fallback to flat config
        elif hasattr(self.config, key):
            return getattr(self.config, key)
        else:
            return default
    
    def _setup_transforms(self):
        """Set up default transforms if not provided"""
        if self.train_transforms is None:
            self.train_transforms = create_train_transforms(
                self.config,
                pathology_aug=self.pathology_aug
            )
        
        if self.val_transforms is None:
            self.val_transforms = create_val_transforms(self.config)
        
        if self.test_transforms is None:
            self.test_transforms = create_test_transforms(self.config)
    
    def setup(self, stage: Optional[str] = None, fold: Optional[int] = None):
        """
        Set up datasets for training/validation/testing
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
            fold: Fold number for k-fold cross-validation
        """
        # Get dataset path
        dataset_path = self._get_config_value('dataset_path', Path('/workspace/DATASET_CLASSIFICATION'))
        cache_dir = self._get_config_value('cache_dir', None)
        
        # Common dataset kwargs
        dataset_kwargs = {
            'modality': self.modality,
            'unified_classes': self.unified_classes,
            'use_cache': self.use_cache,
            'cache_dir': cache_dir,
            'pathology_augmentation': self.pathology_aug,
            'return_metadata': self.return_metadata,
            **self.kwargs
        }
        
        if stage == 'fit' or stage is None:
            if fold is not None:
                # K-fold setup
                self._setup_kfold(dataset_path, dataset_kwargs, fold)
            else:
                # Regular setup
                self._setup_regular(dataset_path, dataset_kwargs)
            
            # Update dataset info
            self._update_dataset_info()
            
            # Log dataset statistics
            self._log_dataset_stats()
        
        if stage == 'test' or stage is None:
            # Test dataset
            if (dataset_path / 'test').exists() or (dataset_path / 'fundus' / 'test').exists():
                self.test_dataset = create_dataset(
                    self.dataset_name,
                    root=dataset_path,
                    split='test',
                    transform=self.test_transforms,
                    **dataset_kwargs
                )
                logger.info(f"Test dataset: {len(self.test_dataset)} samples")
    
    def _setup_regular(self, dataset_path: Path, dataset_kwargs: Dict):
        """Setup regular train/val split"""
        # Training dataset
        self.train_dataset = create_dataset(
            self.dataset_name,
            root=dataset_path,
            split='train',
            transform=self.train_transforms,
            **dataset_kwargs
        )
        
        # Validation dataset
        # Check if val split exists
        val_exists = False
        if self.modality == 'both':
            val_exists = (dataset_path / 'fundus' / 'val').exists() or (dataset_path / 'oct' / 'val').exists()
        elif self.modality == 'fundus':
            val_exists = (dataset_path / 'fundus' / 'val').exists()
        else:  # oct
            val_exists = (dataset_path / 'oct' / 'val').exists()
        
        val_split = 'val' if val_exists else 'test'
        
        self.val_dataset = create_dataset(
            self.dataset_name,
            root=dataset_path,
            split=val_split,
            transform=self.val_transforms,
            **dataset_kwargs
        )
        
        logger.info(
            f"Setup complete - Train: {len(self.train_dataset):,}, "
            f"Val: {len(self.val_dataset):,} (using {val_split} split)"
        )
    
    def _setup_kfold(self, dataset_path: Path, dataset_kwargs: Dict, fold: int):
        """Setup k-fold cross-validation"""
        if self.fold_indices is None:
            # Create full dataset
            full_dataset = create_dataset(
                self.dataset_name,
                root=dataset_path,
                split='train',  # Use train split for k-fold
                transform=None,  # Transform applied later
                **dataset_kwargs
            )
            
            # Generate k-fold indices
            self.n_folds = self._get_config_value('n_folds', 5)
            self.fold_indices = self._create_stratified_kfold(
                full_dataset, self.n_folds
            )
        
        self.current_fold = fold
        train_indices, val_indices = self.fold_indices[fold]
        
        # Create subset datasets
        full_dataset_train = create_dataset(
            self.dataset_name,
            root=dataset_path,
            split='train',
            transform=self.train_transforms,
            **dataset_kwargs
        )
        
        full_dataset_val = create_dataset(
            self.dataset_name,
            root=dataset_path,
            split='train',  # Same split, different indices
            transform=self.val_transforms,
            **dataset_kwargs
        )
        
        self.train_dataset = Subset(full_dataset_train, train_indices)
        self.val_dataset = Subset(full_dataset_val, val_indices)
        
        logger.info(
            f"K-Fold {fold+1}/{self.n_folds} - "
            f"Train: {len(self.train_dataset):,}, "
            f"Val: {len(self.val_dataset):,}"
        )
    
    def _create_stratified_kfold(
        self, 
        dataset: Dataset, 
        n_folds: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified k-fold splits"""
        from sklearn.model_selection import StratifiedKFold
        
        # Get all labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_indices = []
        
        for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
            fold_indices.append((train_idx, val_idx))
        
        return fold_indices
    
    def _update_dataset_info(self):
        """Update dataset information from train dataset"""
        if self.train_dataset is None:
            return
        
        # Handle Subset for k-fold
        dataset = self.train_dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        
        # Get number of classes and names based on configuration
        if self.unified_classes:
            self.num_classes = NUM_TOTAL_CLASSES
            self.class_names = UNIFIED_CLASS_NAMES
        else:
            if self.modality == 'fundus':
                self.num_classes = NUM_FUNDUS_CLASSES
                self.class_names = FUNDUS_CLASS_NAMES
            elif self.modality == 'oct':
                self.num_classes = NUM_OCT_CLASSES
                self.class_names = OCT_CLASS_NAMES
            else:  # both but not unified
                self.num_classes = NUM_FUNDUS_CLASSES + NUM_OCT_CLASSES
                self.class_names = FUNDUS_CLASS_NAMES + OCT_CLASS_NAMES
        
        # Update config if needed
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'num_classes'):
            if self.config.model.num_classes != self.num_classes:
                logger.warning(
                    f"Updating model.num_classes from {self.config.model.num_classes} "
                    f"to {self.num_classes}"
                )
                self.config.model.num_classes = self.num_classes
        elif hasattr(self.config, 'num_classes'):
            if self.config.num_classes != self.num_classes:
                logger.warning(
                    f"Updating num_classes from {self.config.num_classes} "
                    f"to {self.num_classes}"
                )
                self.config.num_classes = self.num_classes
        
        # Calculate class weights for v6.1
        self.class_weights = self._calculate_class_weights_v61(dataset)
    
    def _calculate_class_weights_v61(self, dataset: Dataset) -> torch.Tensor:
        """Calculate class weights based on v6.1 configuration"""
        # Start with uniform weights
        weights = torch.ones(self.num_classes)
        
        # Apply v6.1 specific weights for minority classes
        if self.unified_classes:
            # Map class names to indices
            minority_weights = {
                22: 2.0,  # ERM (OCT class 4 -> unified 18+4)
                25: 2.0,  # RVO_OCT (OCT class 7 -> unified 18+7)
                27: 1.5,  # RAO_OCT (OCT class 9 -> unified 18+9)
                12: 1.5,  # Myopia_Degenerative (Fundus class 12)
            }
            
            for idx, weight in minority_weights.items():
                if idx < self.num_classes:
                    weights[idx] = weight
        
        # Get actual class distribution if available
        if hasattr(dataset, 'get_class_weights'):
            dataset_weights = dataset.get_class_weights()
            # Combine with v6.1 weights
            weights = weights * dataset_weights
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights
    
    def _log_dataset_stats(self):
        """Log dataset v6.1 statistics"""
        logger.info("\n" + "="*60)
        logger.info("DATASET V6.1 STATISTICS")
        logger.info("="*60)
        logger.info(f"Modality: {self.modality}")
        logger.info(f"Unified classes: {self.unified_classes}")
        logger.info(f"Number of classes: {self.num_classes}")
        
        if self.train_dataset:
            logger.info(f"Training samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            logger.info(f"Validation samples: {len(self.val_dataset):,}")
        
        # Log class imbalance info
        if hasattr(self.train_dataset, 'get_imbalance_ratio'):
            dataset = self.train_dataset
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
            imbalance_ratio = dataset.get_imbalance_ratio()
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # Log minority classes
        logger.info("\nMinority classes with increased weights:")
        if self.unified_classes:
            logger.info("  - ERM (04): weight 2.0")
            logger.info("  - RVO_OCT (07): weight 2.0")
            logger.info("  - RAO_OCT (09): weight 1.5")
            logger.info("  - Myopia_Degenerative (12): weight 1.5")
        
        logger.info("="*60 + "\n")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with v6.1 optimizations"""
        if self.train_dataloader_obj is not None:
            return self.train_dataloader_obj
        
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        # Get batch size
        batch_size = self._get_config_value('batch_size', 16)
        
        # Create sampler if needed
        sampler = None
        shuffle = True
        
        if self.balance_dataset or self.adaptive_sampling:
            # Use class weights for balanced sampling
            sampler = create_sampler(
                dataset=self.train_dataset,
                balanced=self.balance_dataset,
                adaptive=self.adaptive_sampling,
                class_weights=self.class_weights
            )
            shuffle = False  # Sampler handles shuffling
        
        # Get dataloader settings
        num_workers = self._get_config_value('num_workers', 8)
        pin_memory = self._get_config_value('pin_memory', True)
        persistent_workers = self._get_config_value('persistent_workers', True)
        prefetch_factor = self._get_config_value('prefetch_factor', 2)
        
        # Create DataLoader
        self.train_dataloader_obj = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            drop_last=True,
            collate_fn=self._collate_fn if self.return_metadata else None
        )
        
        return self.train_dataloader_obj
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader"""
        if self.val_dataloader_obj is not None:
            return self.val_dataloader_obj
        
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not initialized. Call setup() first.")
        
        # Get settings
        batch_size = self._get_config_value('batch_size', 16)
        val_batch_size = batch_size * 2  # Larger batch for validation
        num_workers = self._get_config_value('num_workers', 8)
        pin_memory = self._get_config_value('pin_memory', True)
        persistent_workers = self._get_config_value('persistent_workers', True)
        
        # Create DataLoader
        self.val_dataloader_obj = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
            collate_fn=self._collate_fn if self.return_metadata else None
        )
        
        return self.val_dataloader_obj
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader"""
        if self.test_dataloader_obj is not None:
            return self.test_dataloader_obj
        
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        
        # Get settings
        batch_size = self._get_config_value('batch_size', 16)
        test_batch_size = batch_size * 2
        num_workers = self._get_config_value('num_workers', 8)
        pin_memory = self._get_config_value('pin_memory', True)
        persistent_workers = self._get_config_value('persistent_workers', True)
        
        # Create DataLoader
        self.test_dataloader_obj = DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=False,
            collate_fn=self._collate_fn if self.return_metadata else None
        )
        
        return self.test_dataloader_obj
    
    def _collate_fn(self, batch: List[Tuple]) -> Tuple:
        """Custom collate function for batches with metadata"""
        if len(batch[0]) == 2:
            # No metadata
            images, labels = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(labels)
            return images, labels
        else:
            # With metadata
            images, labels, metadata = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(labels)
            
            # Combine metadata
            combined_metadata = {}
            for key in metadata[0].keys():
                combined_metadata[key] = [m[key] for m in metadata]
            
            return images, labels, combined_metadata
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing"""
        # Clear references to help garbage collection
        self.train_dataloader_obj = None
        self.val_dataloader_obj = None
        self.test_dataloader_obj = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics for v6.1"""
        stats = {
            'dataset_version': '6.1',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'modality': self.modality,
            'unified_classes': self.unified_classes,
        }
        
        if self.train_dataset:
            stats['train_size'] = len(self.train_dataset)
            
        if self.val_dataset:
            stats['val_size'] = len(self.val_dataset)
            
        if self.test_dataset:
            stats['test_size'] = len(self.test_dataset)
        
        if self.class_weights is not None:
            stats['class_weights'] = self.class_weights.tolist()
        
        # Add v6.1 specific stats
        stats['dataset_stats'] = DATASET_V61_STATS
        
        # Add k-fold info if applicable
        if self.n_folds:
            stats['k_fold'] = {
                'n_folds': self.n_folds,
                'current_fold': self.current_fold
            }
        
        return stats
    
    def get_sample_batch(self, split: str = 'train', size: int = 8) -> Tuple:
        """Get a sample batch for testing"""
        if split == 'train' and self.train_dataset:
            dataset = self.train_dataset
            transform = self.train_transforms
        elif split == 'val' and self.val_dataset:
            dataset = self.val_dataset
            transform = self.val_transforms
        elif split == 'test' and self.test_dataset:
            dataset = self.test_dataset
            transform = self.test_transforms
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Get random samples
        indices = np.random.choice(len(dataset), size=min(size, len(dataset)), replace=False)
        batch = [dataset[i] for i in indices]
        
        # Collate
        return self._collate_fn(batch) if self._collate_fn else batch


def create_datamodule(
    config: RETFoundConfig,
    dataset_name: str = 'caasi_v61',
    **kwargs
) -> RETFoundDataModule:
    """
    Create a DataModule for dataset v6.1
    
    Args:
        config: Configuration object
        dataset_name: Name of dataset (default: 'caasi_v61')
        **kwargs: Additional arguments
        
    Returns:
        DataModule instance
    """
    return RETFoundDataModule(config, dataset_name, **kwargs)