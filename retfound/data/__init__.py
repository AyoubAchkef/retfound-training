"""
Data Package for RETFound - Enhanced for Dataset v6.1
=====================================================

Handles all data-related operations including datasets, transforms,
sampling, and caching with full support for CAASI dataset v6.1.
"""

from .datasets import (
    RETFoundDataset,
    BaseDataset,
    CAASIDatasetV61,  # New v6.1 dataset class
    create_dataset,
    list_datasets,
    get_dataset_info
)

from .datamodule import (
    RETFoundDataModule,
    create_datamodule
)

from .transforms import (
    create_train_transforms,
    create_val_transforms,
    create_test_transforms,
    create_advanced_transforms,
    PathologyAugmentation,
    MixupCutmixTransform,
    get_class_augmentation_weight  # New v6.1 function
)

from .samplers import (
    BalancedSampler,
    WeightedSampler,  # Added for v6.1
    AdaptiveSampler,
    create_sampler
)

from .cache import (
    DatasetCache,
    ImageCache,
    get_dataset_cache,  # New function
    get_image_cache,    # New function
    get_cache_stats,    # New function
    cache_v61_dataset_info,  # New v6.1 specific function
    clear_cache
)

__all__ = [
    # Datasets
    'RETFoundDataset',
    'BaseDataset',
    'CAASIDatasetV61',
    'create_dataset',
    'list_datasets',
    'get_dataset_info',
    
    # DataModule
    'RETFoundDataModule',
    'create_datamodule',
    
    # Transforms
    'create_train_transforms',
    'create_val_transforms',
    'create_test_transforms',
    'create_advanced_transforms',
    'PathologyAugmentation',
    'MixupCutmixTransform',
    'get_class_augmentation_weight',
    
    # Samplers
    'BalancedSampler',
    'WeightedSampler',
    'AdaptiveSampler',
    'create_sampler',
    
    # Cache
    'DatasetCache',
    'ImageCache',
    'get_dataset_cache',
    'get_image_cache',
    'get_cache_stats',
    'cache_v61_dataset_info',
    'clear_cache'
]