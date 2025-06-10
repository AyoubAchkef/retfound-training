"""
Data Package for RETFound
========================

Handles all data-related operations including datasets, transforms,
sampling, and caching.
"""

from .datasets import (
    RETFoundDataset,
    BaseDataset,
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
    MixupCutmixTransform
)

from .samplers import (
    BalancedSampler,
    AdaptiveSampler,
    create_sampler
)

from .cache import (
    DatasetCache,
    ImageCache,
    clear_cache
)

__all__ = [
    # Datasets
    'RETFoundDataset',
    'BaseDataset',
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
    
    # Samplers
    'BalancedSampler',
    'AdaptiveSampler',
    'create_sampler',
    
    # Cache
    'DatasetCache',
    'ImageCache',
    'clear_cache'
]
