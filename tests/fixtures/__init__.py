"""
Test Fixtures Module
====================

Provides test data and fixtures for the test suite.
"""

from pathlib import Path

# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent

# Sample images directory
SAMPLE_IMAGES_DIR = FIXTURES_DIR / 'sample_images'

# Sample configs directory
SAMPLE_CONFIGS_DIR = FIXTURES_DIR / 'configs'

# Test data paths
TEST_IMAGE_NORMAL = SAMPLE_IMAGES_DIR / 'normal' / 'test_normal.jpg'
TEST_IMAGE_DR = SAMPLE_IMAGES_DIR / 'dr' / 'test_dr.jpg'
TEST_IMAGE_GLAUCOMA = SAMPLE_IMAGES_DIR / 'glaucoma' / 'test_glaucoma.jpg'

# Test configuration files
TEST_CONFIG_MINIMAL = SAMPLE_CONFIGS_DIR / 'minimal.yaml'
TEST_CONFIG_FULL = SAMPLE_CONFIGS_DIR / 'full.yaml'
TEST_CONFIG_DEBUG = SAMPLE_CONFIGS_DIR / 'debug.yaml'

# Class names for testing
TEST_CLASS_NAMES = [
    'Normal_Fundus',
    'DR_Mild',
    'DR_Moderate',
    'DR_Severe',
    'DR_Proliferative',
    'Normal_OCT',
    'AMD',
    'DME_Fundus',
    'ERM',
    'RAO',
    'RVO',
    'VID',
    'CNV',
    'DME_OCT',
    'DRUSEN',
    'Glaucoma_Normal',
    'Glaucoma_Positive',
    'Glaucoma_Suspect',
    'Myopia',
    'Cataract',
    'Retinal_Detachment',
    'Other'
]

# Test model parameters
TEST_MODEL_PARAMS = {
    'minimal': {
        'model_type': 'vit_tiny_patch16_224',
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'num_classes': 3,
        'input_size': 224
    },
    'small': {
        'model_type': 'vit_small_patch16_224',
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'num_classes': 10,
        'input_size': 224
    },
    'full': {
        'model_type': 'vit_large_patch16_224',
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'num_classes': 22,
        'input_size': 224
    }
}

# Test optimization configurations
TEST_OPTIMIZATION_CONFIGS = {
    'basic': {
        'use_sam': False,
        'use_ema': False,
        'use_tta': False,
        'use_amp': False,
        'use_compile': False
    },
    'advanced': {
        'use_sam': True,
        'use_ema': True,
        'use_tta': True,
        'use_amp': True,
        'use_compile': False  # Often causes issues in tests
    }
}

# Sample metrics for testing
SAMPLE_METRICS = {
    'accuracy': 92.5,
    'balanced_accuracy': 89.3,
    'cohen_kappa': 0.856,
    'auc_macro': 0.943,
    'mean_sensitivity': 88.7,
    'mean_specificity': 91.2,
    'DR_Mild_sensitivity': 85.3,
    'DR_Mild_specificity': 92.1,
    'DR_Mild_auc': 0.921,
    'RAO_sensitivity': 98.5,
    'RAO_specificity': 99.1,
    'RAO_auc': 0.995
}


def create_dummy_checkpoint(num_classes=22, epoch=10):
    """Create a dummy checkpoint for testing"""
    import torch
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {},  # Would be populated with actual state dict
        'optimizer_state_dict': {},
        'scheduler_state_dict': {},
        'metrics': SAMPLE_METRICS,
        'history': {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'train_acc': [80.0, 85.0, 90.0],
            'val_acc': [78.0, 83.0, 88.0]
        },
        'best_val_acc': 92.5,
        'best_val_auc': 0.943,
        'best_epoch': 8,
        'config': {
            'num_classes': num_classes,
            'input_size': 224,
            'model_type': 'vit_large_patch16_224'
        },
        'class_names': TEST_CLASS_NAMES[:num_classes]
    }
    
    return checkpoint


def create_sample_image(size=(224, 224, 3), image_type='normal'):
    """Create a sample image for testing"""
    import numpy as np
    
    # Create different patterns for different types
    if image_type == 'normal':
        # Uniform gray image
        image = np.ones(size, dtype=np.uint8) * 128
    elif image_type == 'dr':
        # Add some red spots to simulate hemorrhages
        image = np.ones(size, dtype=np.uint8) * 100
        # Add red channel variations
        image[:, :, 0] = 150
        # Add some spots
        for _ in range(10):
            x, y = np.random.randint(50, size[0]-50), np.random.randint(50, size[1]-50)
            image[x-5:x+5, y-5:y+5, 0] = 200
    elif image_type == 'glaucoma':
        # Create a disc-like pattern
        image = np.ones(size, dtype=np.uint8) * 120
        center = (size[0] // 2, size[1] // 2)
        radius = 50
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        image[mask] = 180
    else:
        # Random noise
        image = np.random.randint(0, 256, size, dtype=np.uint8)
    
    return image


def create_sample_config(config_type='minimal'):
    """Create a sample configuration dictionary"""
    import copy
    
    base_config = {
        'base_path': '/tmp/retfound_test',
        'num_classes': 3,
        'input_size': 224,
        'batch_size': 4,
        'epochs': 2,
        'base_lr': 1e-4,
        'use_amp': False,
        'use_sam': False,
        'use_ema': False,
        'use_tta': False
    }
    
    if config_type == 'full':
        config = copy.deepcopy(base_config)
        config.update({
            'num_classes': 22,
            'batch_size': 16,
            'epochs': 100,
            'base_lr': 5e-5,
            'use_amp': True,
            'use_sam': True,
            'use_ema': True,
            'use_tta': True,
            'use_wandb': True,
            'wandb_project': 'test-project'
        })
    elif config_type == 'debug':
        config = copy.deepcopy(base_config)
        config.update({
            'epochs': 1,
            'log_interval': 1,
            'save_frequency': 1,
            'num_workers': 0,
            'persistent_workers': False
        })
    else:
        config = base_config
    
    return config


# Fixture data for different test scenarios
FIXTURE_DATA = {
    'checkpoints': {
        'basic': create_dummy_checkpoint(num_classes=3, epoch=5),
        'advanced': create_dummy_checkpoint(num_classes=22, epoch=50),
    },
    'images': {
        'normal': create_sample_image(image_type='normal'),
        'dr': create_sample_image(image_type='dr'),
        'glaucoma': create_sample_image(image_type='glaucoma'),
        'random': create_sample_image(image_type='random')
    },
    'configs': {
        'minimal': create_sample_config('minimal'),
        'full': create_sample_config('full'),
        'debug': create_sample_config('debug')
    }
}