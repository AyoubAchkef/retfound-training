"""
Dataset Tests
=============

Test dataset loading, augmentation, and data module functionality.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from retfound.core.config import RETFoundConfig
from retfound.data.datasets import CAASIDatasetV61
from retfound.data.datamodule import RETFoundDataModule
from retfound.data.transforms import (
    create_train_transform, create_eval_transform,
    PathologyAugmentation, get_pathology_specific_augmentation
)
from retfound.data.samplers import BalancedBatchSampler, ClassAwareSampler
from retfound.data.cache import ImageCache


class TestCAASIDatasetV61:
    """Test CAASI Dataset v6.1 class"""
    
    def test_dataset_creation(self, test_data_dir, minimal_config):
        """Test creating dataset"""
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None
        )
        
        # Check dataset properties
        assert len(dataset) > 0
        assert hasattr(dataset, 'data')
        assert hasattr(dataset, 'targets')
        assert hasattr(dataset, 'classes')
        assert dataset.num_classes == 28  # v6.1 unified classes
    
    def test_dataset_getitem(self, test_data_dir, minimal_config):
        """Test getting items from dataset"""
        transform = create_eval_transform(minimal_config)
        
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=transform
        )
        
        # Get an item
        image, label = dataset[0]
        
        # Check types and shapes
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, (int, torch.Tensor))
        assert image.shape == (3, minimal_config.input_size, minimal_config.input_size)
        assert 0 <= label < dataset.num_classes
    
    def test_dataset_modality_filtering(self, test_data_dir, minimal_config):
        """Test modality filtering"""
        # Test fundus only
        fundus_dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='fundus',
            unified_classes=False
        )
        
        # Test OCT only
        oct_dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='oct',
            unified_classes=False
        )
        
        # Check modality filtering
        assert all(mod == 'fundus' for mod in fundus_dataset.modalities)
        assert all(mod == 'oct' for mod in oct_dataset.modalities)
    
    def test_dataset_with_cache(self, test_data_dir, minimal_config, temp_output_dir):
        """Test dataset with caching enabled"""
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None,
            use_cache=True,
            cache_dir=temp_output_dir / 'cache'
        )
        
        # Load same image twice
        img1, _ = dataset[0]
        img2, _ = dataset[0]
        
        # Should be loaded from cache second time
        assert np.array_equal(img1, img2)
    
    def test_dataset_class_weights(self, test_data_dir, minimal_config):
        """Test class weights calculation"""
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both'
        )
        
        weights = dataset.get_class_weights()
        
        assert len(weights) == dataset.num_classes
        assert all(w > 0 for w in weights)


class TestMedicalImageDataset:
    """Test medical image dataset functionality"""
    
    def test_pathology_mapping(self, test_data_dir, minimal_config):
        """Test pathology name mapping"""
        minimal_config.dataset_path = test_data_dir
        
        # Create dataset with medical mappings
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None,
            unified_classes=True
        )
        
        # Check class names are available
        assert hasattr(dataset, 'classes')
        assert len(dataset.classes) > 0
        assert hasattr(dataset, 'class_to_idx')
    
    def test_weighted_sampling(self, test_data_dir):
        """Test weighted sampling for imbalanced data"""
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None
        )
        
        # Get class weights
        weights = dataset.get_class_weights()
        
        assert len(weights) == dataset.num_classes
        assert all(w > 0 for w in weights)
        
        # Check that weights are properly normalized
        assert abs(weights.mean().item() - 1.0) < 0.1


class TestDataModule:
    """Test data module functionality"""
    
    def test_datamodule_setup(self, test_data_dir, minimal_config):
        """Test data module setup"""
        minimal_config.dataset_path = test_data_dir
        
        dm = RETFoundDataModule(minimal_config)
        dm.setup()
        
        # Check datasets created
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert hasattr(dm, 'num_classes')
        assert hasattr(dm, 'class_names')
    
    def test_datamodule_dataloaders(self, test_data_dir, minimal_config):
        """Test creating data loaders"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.batch_size = 2
        minimal_config.num_workers = 0
        
        dm = RETFoundDataModule(minimal_config)
        dm.setup()
        
        # Get dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        
        # Check properties
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Get a batch
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] == minimal_config.batch_size
        assert labels.shape[0] == minimal_config.batch_size
    
    def test_datamodule_kfold(self, test_data_dir, minimal_config):
        """Test k-fold functionality"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.use_kfold = True
        minimal_config.n_folds = 3
        
        dm = RETFoundDataModule(minimal_config)
        
        # Setup for fold 0
        dm.setup(fold=0)
        fold0_train_size = len(dm.train_dataset)
        
        # Setup for fold 1
        dm.setup(fold=1)
        fold1_train_size = len(dm.train_dataset)
        
        # Sizes should be similar but not identical
        assert abs(fold0_train_size - fold1_train_size) < fold0_train_size * 0.1


class TestTransforms:
    """Test image transformations"""
    
    def test_train_transform(self, minimal_config):
        """Test training transforms"""
        transform = create_train_transform(minimal_config)
        
        # Create dummy image
        image = Image.new('RGB', (256, 256), color='red')
        
        # Apply transform
        transformed = transform(image)
        
        # Check output
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, minimal_config.input_size, minimal_config.input_size)
    
    def test_eval_transform(self, minimal_config):
        """Test evaluation transforms"""
        transform = create_eval_transform(minimal_config)
        
        # Create dummy image
        image = Image.new('RGB', (256, 256), color='blue')
        
        # Apply transform
        transformed = transform(image)
        
        # Check output
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, minimal_config.input_size, minimal_config.input_size)
    
    def test_pathology_augmentation(self, minimal_config):
        """Test pathology-specific augmentations"""
        aug = PathologyAugmentation(config=minimal_config)
        
        # Test getting augmentation for different pathologies
        dr_aug = aug.get_augmentation('diabetic_retinopathy')
        glaucoma_aug = aug.get_augmentation('glaucoma')
        
        # Should return different augmentations
        assert dr_aug is not None
        assert glaucoma_aug is not None
        
        # Apply augmentation
        image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        if dr_aug:
            augmented = dr_aug(image=image)
            assert 'image' in augmented
            assert augmented['image'].shape == image.shape


class TestSamplers:
    """Test custom samplers"""
    
    def test_balanced_batch_sampler(self, test_data_dir):
        """Test balanced batch sampler"""
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None
        )
        
        sampler = BalancedBatchSampler(
            dataset=dataset,
            batch_size=6,
            n_classes=3,
            n_samples=2
        )
        
        # Get a batch
        batch_indices = next(iter(sampler))
        
        # Check batch composition
        assert len(batch_indices) == 6
        
        # Each class should appear n_samples times
        labels = [dataset.targets[i] for i in batch_indices]
        from collections import Counter
        label_counts = Counter(labels)
        
        for count in label_counts.values():
            assert count == 2
    
    def test_class_aware_sampler(self, test_data_dir):
        """Test class-aware sampler"""
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None
        )
        
        sampler = ClassAwareSampler(
            dataset=dataset,
            num_samples_per_class=5
        )
        
        # Get all indices
        indices = list(sampler)
        
        # Check total samples
        expected_total = 5 * dataset.num_classes
        assert len(indices) == expected_total
        
        # Check distribution
        labels = [dataset.targets[i] for i in indices]
        from collections import Counter
        label_counts = Counter(labels)
        
        for count in label_counts.values():
            assert count == 5


class TestImageCache:
    """Test image caching functionality"""
    
    def test_cache_store_retrieve(self, temp_output_dir):
        """Test storing and retrieving from cache"""
        cache = ImageCache(cache_dir=temp_output_dir / 'cache')
        
        # Create test image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        key = "test_image_1"
        
        # Store in cache
        cache.put(key, image)
        
        # Retrieve from cache
        cached_image = cache.get(key)
        
        assert cached_image is not None
        assert np.array_equal(image, cached_image)
    
    def test_cache_miss(self, temp_output_dir):
        """Test cache miss behavior"""
        cache = ImageCache(cache_dir=temp_output_dir / 'cache')
        
        # Try to get non-existent image
        result = cache.get("non_existent_key")
        
        assert result is None
    
    def test_cache_size_limit(self, temp_output_dir):
        """Test cache size limiting"""
        cache = ImageCache(
            cache_dir=temp_output_dir / 'cache',
            max_size_gb=0.0001  # Very small limit
        )
        
        # Add multiple images
        for i in range(10):
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cache.put(f"image_{i}", image)
        
        # Cache should have evicted some images
        cache_stats = cache.get_stats()
        assert cache_stats['current_size_mb'] < 0.1  # Less than limit


class TestDataUtils:
    """Test data utility functions"""
    
    def test_compute_mean_std(self, test_data_dir):
        """Test computing dataset mean and std"""
        from retfound.data.utils import compute_dataset_stats
        
        stats = compute_dataset_stats(
            data_dir=test_data_dir / 'train',
            num_samples=10
        )
        
        assert 'mean' in stats
        assert 'std' in stats
        assert len(stats['mean']) == 3
        assert len(stats['std']) == 3
        assert all(0 <= m <= 1 for m in stats['mean'])
        assert all(0 <= s <= 1 for s in stats['std'])
    
    def test_split_dataset(self, test_data_dir):
        """Test dataset splitting"""
        from retfound.data.utils import split_dataset
        
        dataset = CAASIDatasetV61(
            root=test_data_dir,
            split='train',
            modality='both',
            transform=None
        )
        
        # Split dataset
        train_dataset, val_dataset = split_dataset(
            dataset,
            val_split=0.2,
            stratify=True
        )
        
        # Check sizes
        total_size = len(dataset)
        assert len(train_dataset) + len(val_dataset) == total_size
        assert abs(len(val_dataset) / total_size - 0.2) < 0.05
        
        # Check stratification
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
        
        from collections import Counter
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        
        # All classes should be present in both splits
        assert set(train_dist.keys()) == set(val_dist.keys())
