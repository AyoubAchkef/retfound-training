"""
Pytest Configuration and Fixtures
=================================

Global fixtures for the test suite.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from retfound.core.config import RETFoundConfig
from retfound.models.retfound import RETFoundModel


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            for class_idx in range(3):  # 3 test classes
                class_dir = test_dir / split / f'{class_idx}_class{class_idx}'
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                n_images = 10 if split == 'train' else 3
                for i in range(n_images):
                    img = Image.new('RGB', (224, 224), color=(class_idx*80, 100, 150))
                    img.save(class_dir / f'image_{i}.jpg')
        
        yield test_dir


@pytest.fixture(scope="session")
def sample_checkpoint(test_data_dir):
    """Create a sample checkpoint file"""
    checkpoint_path = test_data_dir / "test_checkpoint.pth"
    
    # Create minimal checkpoint
    config = RETFoundConfig(
        num_classes=3,
        epochs=5,
        batch_size=2,
        dataset_path=test_data_dir
    )
    
    model = RETFoundModel(config)
    
    checkpoint = {
        'epoch': 2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'config': config.to_dict(),
        'metrics': {
            'accuracy': 85.5,
            'auc_macro': 0.92
        },
        'best_val_acc': 85.5,
        'best_val_auc': 0.92,
        'class_names': ['Class_0', 'Class_1', 'Class_2']
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    yield checkpoint_path
    
    # Cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()


@pytest.fixture
def minimal_config():
    """Minimal configuration for testing"""
    return RETFoundConfig(
        num_classes=3,
        epochs=2,
        batch_size=2,
        input_size=224,
        base_lr=1e-4,
        use_amp=False,  # Disable for CPU testing
        use_compile=False,
        use_gradient_checkpointing=False,
        use_wandb=False,
        use_tensorboard=False
    )


@pytest.fixture
def sample_model(minimal_config):
    """Create a sample model for testing"""
    model = RETFoundModel(minimal_config)
    return model


@pytest.fixture
def sample_batch():
    """Create a sample batch of data"""
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.tensor([0, 1])
    return images, labels


@pytest.fixture
def device():
    """Get test device (CPU for testing)"""
    return torch.device('cpu')


@pytest.fixture
def sample_weights_file(test_data_dir):
    """Create a sample weights file mimicking RETFound format"""
    weights_path = test_data_dir / "retfound_test_weights.pth"
    
    # Create a minimal state dict that looks like RETFound MAE
    state_dict = {
        'encoder.cls_token': torch.randn(1, 1, 1024),
        'encoder.pos_embed': torch.randn(1, 197, 1024),
        'encoder.patch_embed.proj.weight': torch.randn(1024, 3, 16, 16),
        'encoder.patch_embed.proj.bias': torch.randn(1024),
        'encoder.blocks.0.norm1.weight': torch.randn(1024),
        'encoder.blocks.0.norm1.bias': torch.randn(1024),
        'encoder.blocks.0.attn.qkv.weight': torch.randn(3072, 1024),
        'encoder.blocks.0.attn.qkv.bias': torch.randn(3072),
        'encoder.blocks.0.attn.proj.weight': torch.randn(1024, 1024),
        'encoder.blocks.0.attn.proj.bias': torch.randn(1024),
        'encoder.norm.weight': torch.randn(1024),
        'encoder.norm.bias': torch.randn(1024),
    }
    
    checkpoint = {
        'model': state_dict,
        'epoch': 100,
        'model_state': state_dict  # Some checkpoints use this key
    }
    
    torch.save(checkpoint, weights_path)
    
    yield weights_path
    
    # Cleanup
    if weights_path.exists():
        weights_path.unlink()


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary"""
    return {
        'accuracy': 92.5,
        'balanced_accuracy': 91.8,
        'cohen_kappa': 0.88,
        'auc_macro': 0.95,
        'mean_sensitivity': 90.2,
        'mean_specificity': 93.4,
        'Class_0_sensitivity': 88.5,
        'Class_0_specificity': 94.2,
        'Class_0_auc': 0.93,
        'Class_1_sensitivity': 91.2,
        'Class_1_specificity': 92.8,
        'Class_1_auc': 0.96,
        'Class_2_sensitivity': 90.8,
        'Class_2_specificity': 93.2,
        'Class_2_auc': 0.94
    }


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb for testing"""
    class MockWandb:
        run = None
        
        @staticmethod
        def init(**kwargs):
            MockWandb.run = True
            return MockWandb
        
        @staticmethod
        def log(data):
            pass
        
        @staticmethod
        def finish():
            MockWandb.run = None
        
        @staticmethod
        def watch(model, **kwargs):
            pass
    
    monkeypatch.setattr("wandb.init", MockWandb.init)
    monkeypatch.setattr("wandb.log", MockWandb.log)
    monkeypatch.setattr("wandb.finish", MockWandb.finish)
    monkeypatch.setattr("wandb.watch", MockWandb.watch)
    monkeypatch.setattr("wandb.run", MockWandb.run)
    
    return MockWandb


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_files(temp_output_dir):
    """Create sample image files for testing"""
    image_dir = temp_output_dir / "images"
    image_dir.mkdir()
    
    image_paths = []
    for i in range(5):
        img = Image.new('RGB', (512, 512), color=(i*50, 100, 150))
        img_path = image_dir / f'test_image_{i}.jpg'
        img.save(img_path)
        image_paths.append(img_path)
    
    return image_paths


@pytest.fixture
def cleanup_gpu():
    """Cleanup GPU memory after tests"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if needed"""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Test utilities
def assert_tensor_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert two tensors are close"""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"Tensors not close:\nActual:\n{actual}\nExpected:\n{expected}"


def create_dummy_dataset(root_dir, num_classes=5, images_per_class=10):
    """Create a dummy dataset for testing"""
    for split in ['train', 'val', 'test']:
        split_dir = root_dir / split
        for class_idx in range(num_classes):
            class_dir = split_dir / f'{class_idx}_class_{class_idx}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_idx in range(images_per_class):
                # Create random image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(class_dir / f'img_{img_idx}.jpg')


def get_test_device():
    """Get device for testing (prefer CPU for consistency)"""
    return torch.device('cpu')
