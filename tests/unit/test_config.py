"""
Configuration Tests
===================

Test configuration loading, saving, and validation.
"""

import pytest
from pathlib import Path
import yaml
import torch

from retfound.core.config import RETFoundConfig


class TestRETFoundConfig:
    """Test configuration class"""
    
    def test_default_config_creation(self):
        """Test creating default configuration"""
        config = RETFoundConfig()
        
        # Check default values
        assert config.model_type == "vit_large_patch16_224"
        assert config.num_classes == 22
        assert config.batch_size == 16
        assert config.epochs == 100
        assert config.base_lr == 5e-5
        assert config.use_sam == True
        assert config.use_ema == True
    
    def test_config_paths_initialization(self):
        """Test path initialization"""
        config = RETFoundConfig()
        
        # Check paths are initialized
        assert config.dataset_path is not None
        assert config.output_path is not None
        assert config.checkpoint_path is not None
        assert config.cache_dir is not None
        
        # Check paths are Path objects
        assert isinstance(config.dataset_path, Path)
        assert isinstance(config.output_path, Path)
    
    def test_config_save_load(self, temp_output_dir):
        """Test saving and loading configuration"""
        # Create config with custom values
        config = RETFoundConfig(
            num_classes=10,
            batch_size=32,
            epochs=50,
            base_lr=1e-4,
            use_sam=False
        )
        
        # Save config
        config_path = temp_output_dir / "test_config.yaml"
        config.save(config_path)
        
        # Check file exists
        assert config_path.exists()
        
        # Load config
        loaded_config = RETFoundConfig.load(config_path)
        
        # Check values match
        assert loaded_config.num_classes == 10
        assert loaded_config.batch_size == 32
        assert loaded_config.epochs == 50
        assert loaded_config.base_lr == 1e-4
        assert loaded_config.use_sam == False
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = RETFoundConfig(num_classes=5)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['num_classes'] == 5
        assert 'model_type' in config_dict
        assert 'batch_size' in config_dict
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'num_classes': 7,
            'batch_size': 64,
            'epochs': 200,
            'use_mixup': False
        }
        
        config = RETFoundConfig.from_dict(config_dict)
        
        assert config.num_classes == 7
        assert config.batch_size == 64
        assert config.epochs == 200
        assert config.use_mixup == False
    
    def test_config_copy(self):
        """Test copying configuration"""
        config = RETFoundConfig(num_classes=15)
        config_copy = config.copy()
        
        # Check it's a different object
        assert config is not config_copy
        
        # Check values are the same
        assert config_copy.num_classes == config.num_classes
        assert config_copy.batch_size == config.batch_size
        
        # Modify copy shouldn't affect original
        config_copy.num_classes = 20
        assert config.num_classes == 15
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid values
        with pytest.raises(ValueError):
            RETFoundConfig(num_classes=0)
        
        with pytest.raises(ValueError):
            RETFoundConfig(batch_size=-1)
        
        with pytest.raises(ValueError):
            RETFoundConfig(epochs=0)
        
        with pytest.raises(ValueError):
            RETFoundConfig(base_lr=-0.1)
    
    def test_amp_dtype_selection(self):
        """Test automatic AMP dtype selection"""
        config = RETFoundConfig()
        
        # Should be bfloat16 if available, else float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            assert config.amp_dtype == torch.bfloat16
        else:
            assert config.amp_dtype == torch.float16
    
    def test_config_update(self):
        """Test updating configuration"""
        config = RETFoundConfig()
        
        # Update multiple values
        updates = {
            'num_classes': 30,
            'batch_size': 48,
            'use_tta': False
        }
        
        config.update(updates)
        
        assert config.num_classes == 30
        assert config.batch_size == 48
        assert config.use_tta == False
    
    def test_weights_paths_initialization(self):
        """Test weights paths are properly initialized"""
        config = RETFoundConfig()
        
        assert 'cfp' in config.weights_paths
        assert 'oct' in config.weights_paths
        assert 'meh' in config.weights_paths
        
        # Check paths are Path objects
        for key, path in config.weights_paths.items():
            assert isinstance(path, Path)
    
    def test_yaml_serialization(self, temp_output_dir):
        """Test YAML serialization handles Path objects"""
        config = RETFoundConfig()
        config.dataset_path = Path("/custom/dataset/path")
        
        # Save and load
        config_path = temp_output_dir / "path_test.yaml"
        config.save(config_path)
        
        # Load raw YAML to check format
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Paths should be strings in YAML
        assert isinstance(yaml_data['dataset_path'], str)
        
        # Load back should convert to Path
        loaded_config = RETFoundConfig.load(config_path)
        assert isinstance(loaded_config.dataset_path, Path)
        assert str(loaded_config.dataset_path) == "/custom/dataset/path"
    
    def test_device_aware_config(self):
        """Test configuration adapts to available devices"""
        config = RETFoundConfig()
        
        if torch.cuda.is_available():
            assert config.use_amp == True
            assert config.cudnn_benchmark == True
        
        # Compile should only be enabled for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            assert config.use_compile == True
        else:
            assert config.use_compile == False


class TestConfigValidation:
    """Test configuration validation methods"""
    
    def test_validate_numeric_ranges(self):
        """Test numeric parameter validation"""
        config = RETFoundConfig()
        
        # Test valid ranges
        config.validate()  # Should not raise
        
        # Test invalid ranges
        config.layer_decay = 1.5  # Should be between 0 and 1
        with pytest.raises(ValueError):
            config.validate()
        
        config.layer_decay = 0.65  # Reset
        config.drop_path_rate = -0.1  # Should be >= 0
        with pytest.raises(ValueError):
            config.validate()
    
    def test_validate_paths_exist(self):
        """Test path existence validation"""
        config = RETFoundConfig()
        
        # Set non-existent path
        config.dataset_path = Path("/non/existent/path")
        
        # Should warn but not fail
        config.validate(check_paths=True)  # Should log warning
    
    def test_validate_dependencies(self):
        """Test configuration dependency validation"""
        config = RETFoundConfig()
        
        # If using SAM, certain parameters should be set
        config.use_sam = True
        config.sam_rho = 0.0  # Invalid for SAM
        
        with pytest.raises(ValueError):
            config.validate()
        
        # If using focal loss, alpha should be set
        config.use_focal_loss = True
        config.focal_alpha = None
        
        with pytest.warns(UserWarning):
            config.validate()
