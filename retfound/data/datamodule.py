"""
DataModule for RETFound
======================

Lightning-style DataModule for managing data loading, transforms,
and dataset splits.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, List

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..core.config import RETFoundConfig
from .datasets import create_dataset, RETFoundDataset
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
    DataModule for RETFound training
    
    Handles:
    - Dataset creation and splitting
    - Transform application
    - DataLoader creation
    - Sampling strategies
    """
    
    def __init__(
        self,
        config: RETFoundConfig,
        dataset_name: str = 'retfound',
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize DataModule
        
        Args:
            config: Configuration object
            dataset_name: Name of dataset to use
            train_transforms: Custom train transforms
            val_transforms: Custom validation transforms
            test_transforms: Custom test transforms
            **kwargs: Additional arguments passed to dataset
        """
        self.config = config
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        
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
        if config.use_pathology_augmentation:
            self.pathology_aug = PathologyAugmentation(config.input_size)
        
        # Setup transforms if not provided
        self._setup_transforms()
        
        # Dataset info
        self.num_classes: Optional[int] = None
        self.class_names: Optional[List[str]] = None
        self.class_weights: Optional[torch.Tensor] = None
        
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
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training/validation/testing
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        # Common dataset kwargs
        dataset_kwargs = {
            'use_cache': True,
            'cache_dir': self.config.cache_dir,
            'pathology_augmentation': self.pathology_aug,
            **self.kwargs
        }
        
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = create_dataset(
                self.dataset_name,
                root=self.config.dataset_path,
                split='train',
                transform=self.train_transforms,
                **dataset_kwargs
            )
            
            # Validation dataset
            val_split = 'val' if (self.config.dataset_path / 'val').exists() else 'test'
            self.val_dataset = create_dataset(
                self.dataset_name,
                root=self.config.dataset_path,
                split=val_split,
                transform=self.val_transforms,
                **dataset_kwargs
            )
            
            # Update dataset info
            self._update_dataset_info()
            
            logger.info(
                f"Setup complete - Train: {len(self.train_dataset)}, "
                f"Val: {len(self.val_dataset)}"
            )
        
        if stage == 'test' or stage is None:
            # Test dataset
            if (self.config.dataset_path / 'test').exists():
                self.test_dataset = create_dataset(
                    self.dataset_name,
                    root=self.config.dataset_path,
                    split='test',
                    transform=self.test_transforms,
                    **dataset_kwargs
                )
                logger.info(f"Test dataset: {len(self.test_dataset)} samples")
    
    def _update_dataset_info(self):
        """Update dataset information from train dataset"""
        if self.train_dataset is None:
            return
        
        # Get number of classes
        if hasattr(self.train_dataset, 'classes'):
            self.num_classes = len(self.train_dataset.classes)
            self.class_names = self.train_dataset.classes
        else:
            # Infer from targets
            targets = self.train_dataset.get_labels()
            self.num_classes = len(set(targets))
            self.class_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        # Update config if needed
        if self.config.num_classes != self.num_classes:
            logger.warning(
                f"Updating num_classes from {self.config.num_classes} "
                f"to {self.num_classes}"
            )
            self.config.num_classes = self.num_classes
        
        # Calculate class weights
        if hasattr(self.train_dataset, 'get_class_weights'):
            self.class_weights = self.train_dataset.get_class_weights()
        
        # Log class distribution
        if hasattr(self.train_dataset, 'get_imbalance_ratio'):
            imbalance_ratio = self.train_dataset.get_imbalance_ratio()
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader"""
        if self.train_dataloader_obj is not None:
            return self.train_dataloader_obj
        
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        # Create sampler if needed
        sampler = None
        shuffle = True
        
        if self.config.balance_dataset or self.config.adaptive_sampling:
            sampler = create_sampler(
                dataset=self.train_dataset,
                balanced=self.config.balance_dataset,
                adaptive=self.config.adaptive_sampling
            )
            shuffle = False  # Sampler handles shuffling
        
        # Create DataLoader
        self.train_dataloader_obj = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True
        )
        
        return self.train_dataloader_obj
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader"""
        if self.val_dataloader_obj is not None:
            return self.val_dataloader_obj
        
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not initialized. Call setup() first.")
        
        # Create DataLoader
        self.val_dataloader_obj = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=False
        )
        
        return self.val_dataloader_obj
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader"""
        if self.test_dataloader_obj is not None:
            return self.test_dataloader_obj
        
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        
        # Create DataLoader
        self.test_dataloader_obj = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=False
        )
        
        return self.test_dataloader_obj
    
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
        """Get dataset statistics"""
        stats = {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
        }
        
        if self.train_dataset:
            stats['train_size'] = len(self.train_dataset)
            
        if self.val_dataset:
            stats['val_size'] = len(self.val_dataset)
            
        if self.test_dataset:
            stats['test_size'] = len(self.test_dataset)
        
        if self.class_weights is not None:
            stats['class_weights'] = self.class_weights.tolist()
        
        return stats


def create_datamodule(
    config: RETFoundConfig,
    dataset_name: str = 'retfound',
    **kwargs
) -> RETFoundDataModule:
    """
    Create a DataModule
    
    Args:
        config: Configuration object
        dataset_name: Name of dataset
        **kwargs: Additional arguments
        
    Returns:
        DataModule instance
    """
    return RETFoundDataModule(config, dataset_name, **kwargs)
