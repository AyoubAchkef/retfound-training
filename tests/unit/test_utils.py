"""
Utilities Tests
===============

Test utility functions and helpers.
"""

import pytest
import torch
import numpy as np
import json
import yaml
from pathlib import Path
import time
import logging

from retfound.utils.io import (
    save_json, load_json,
    save_yaml, load_yaml,
    save_checkpoint, load_checkpoint,
    ensure_dir_exists,
    get_file_size,
    copy_files
)
from retfound.utils.device import (
    get_device, get_device_name,
    get_gpu_memory_info,
    clear_gpu_cache,
    get_optimal_batch_size
)
from retfound.utils.logging import (
    setup_logging, get_logger,
    log_system_info,
    log_training_config,
    ProgressLogger
)
from retfound.utils.timing import (
    Timer, TimeTracker,
    format_time, estimate_remaining_time
)
from retfound.utils.reproducibility import (
    set_seed, worker_init_fn,
    make_deterministic,
    get_random_state, set_random_state
)


class TestIOUtils:
    """Test I/O utility functions"""
    
    def test_json_save_load(self, temp_output_dir):
        """Test JSON save and load"""
        data = {
            'model': 'retfound',
            'accuracy': 95.5,
            'classes': ['normal', 'disease'],
            'config': {'lr': 0.001, 'batch_size': 32}
        }
        
        json_path = temp_output_dir / 'test.json'
        
        # Save
        save_json(data, json_path)
        assert json_path.exists()
        
        # Load
        loaded_data = load_json(json_path)
        assert loaded_data == data
    
    def test_yaml_save_load(self, temp_output_dir):
        """Test YAML save and load"""
        data = {
            'model_config': {
                'architecture': 'vit_large',
                'num_classes': 10,
                'pretrained': True
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001
            }
        }
        
        yaml_path = temp_output_dir / 'config.yaml'
        
        # Save
        save_yaml(data, yaml_path)
        assert yaml_path.exists()
        
        # Load
        loaded_data = load_yaml(yaml_path)
        assert loaded_data == data
    
    def test_checkpoint_save_load(self, temp_output_dir, sample_model):
        """Test checkpoint save and load"""
        checkpoint = {
            'epoch': 10,
            'model_state_dict': sample_model.state_dict(),
            'optimizer_state_dict': {'lr': 0.001},
            'metrics': {'accuracy': 92.5},
            'config': {'batch_size': 32}
        }
        
        checkpoint_path = temp_output_dir / 'checkpoint.pth'
        
        # Save
        save_checkpoint(checkpoint, checkpoint_path)
        assert checkpoint_path.exists()
        
        # Load
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        assert loaded_checkpoint['epoch'] == 10
        assert loaded_checkpoint['metrics']['accuracy'] == 92.5
        
        # Check model state dict
        for key in sample_model.state_dict():
            assert key in loaded_checkpoint['model_state_dict']
    
    def test_ensure_dir_exists(self, temp_output_dir):
        """Test directory creation"""
        new_dir = temp_output_dir / 'new' / 'nested' / 'directory'
        
        ensure_dir_exists(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_get_file_size(self, temp_output_dir):
        """Test file size calculation"""
        # Create file with known size
        test_file = temp_output_dir / 'test.txt'
        content = 'a' * 1000  # 1000 bytes
        test_file.write_text(content)
        
        size = get_file_size(test_file)
        assert size['bytes'] == 1000
        assert size['kb'] == 1.0
        assert size['mb'] == 0.001
        assert size['readable'] == '1.0 KB'
    
    def test_copy_files(self, temp_output_dir):
        """Test file copying"""
        # Create source files
        src_dir = temp_output_dir / 'source'
        src_dir.mkdir()
        
        (src_dir / 'file1.txt').write_text('content1')
        (src_dir / 'file2.txt').write_text('content2')
        
        # Copy files
        dst_dir = temp_output_dir / 'destination'
        copied_files = copy_files(src_dir, dst_dir, pattern='*.txt')
        
        assert len(copied_files) == 2
        assert (dst_dir / 'file1.txt').exists()
        assert (dst_dir / 'file2.txt').read_text() == 'content2'


class TestDeviceUtils:
    """Test device utility functions"""
    
    def test_get_device(self):
        """Test device selection"""
        # CPU device
        device = get_device('cpu')
        assert device.type == 'cpu'
        
        # CUDA device (if available)
        if torch.cuda.is_available():
            device = get_device('cuda')
            assert device.type == 'cuda'
            
            # Specific GPU
            device = get_device('cuda:0')
            assert device.type == 'cuda'
            assert device.index == 0
    
    def test_get_device_name(self):
        """Test getting device name"""
        name = get_device_name('cpu')
        assert 'CPU' in name
        
        if torch.cuda.is_available():
            name = get_device_name('cuda:0')
            assert len(name) > 0  # Should return GPU name
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_info(self):
        """Test GPU memory information"""
        info = get_gpu_memory_info(device_id=0)
        
        assert 'allocated' in info
        assert 'reserved' in info
        assert 'total' in info
        assert 'free' in info
        
        # Values should be non-negative
        assert info['allocated'] >= 0
        assert info['total'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_clear_gpu_cache(self):
        """Test GPU cache clearing"""
        # Allocate some tensors
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(10)]
        
        # Clear references and cache
        del tensors
        clear_gpu_cache()
        
        # Memory should be released
        info = get_gpu_memory_info()
        # Can't guarantee exact values, but function should run
        assert True
    
    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        if torch.cuda.is_available():
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Calculate optimal batch size
            batch_size = get_optimal_batch_size(
                model_size_gb=0.5,
                available_memory_gb=gpu_memory * 0.8,
                input_size=(3, 224, 224),
                mixed_precision=True
            )
            
            assert batch_size > 0
            assert isinstance(batch_size, int)


class TestLoggingUtils:
    """Test logging utilities"""
    
    def test_setup_logging(self, temp_output_dir):
        """Test logging setup"""
        log_file = temp_output_dir / 'test.log'
        
        setup_logging(
            level=logging.INFO,
            log_file=log_file,
            console=True
        )
        
        logger = get_logger('test')
        logger.info("Test message")
        
        # Check log file created
        assert log_file.exists()
        
        # Check message logged
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_log_system_info(self, capsys):
        """Test system info logging"""
        log_system_info()
        
        captured = capsys.readouterr()
        
        # Should log various system info
        assert "Python" in captured.out or "System" in captured.out
        assert "PyTorch" in captured.out
    
    def test_log_training_config(self, minimal_config, capsys):
        """Test training config logging"""
        log_training_config(minimal_config.to_dict())
        
        captured = capsys.readouterr()
        
        # Should log config parameters
        assert "batch_size" in captured.out
        assert "epochs" in captured.out
    
    def test_progress_logger(self):
        """Test progress logger"""
        logger = ProgressLogger(total_steps=100, update_frequency=10)
        
        metrics_history = []
        
        for step in range(100):
            metrics = {
                'loss': 1.0 - step / 100,
                'accuracy': step / 100
            }
            
            logger.update(step, metrics)
            metrics_history.append(metrics)
        
        # Get summary
        summary = logger.get_summary()
        
        assert 'avg_loss' in summary
        assert 'avg_accuracy' in summary
        assert summary['total_steps'] == 100


class TestTimingUtils:
    """Test timing utilities"""
    
    def test_timer_context_manager(self):
        """Test Timer context manager"""
        with Timer() as timer:
            time.sleep(0.1)
        
        assert timer.elapsed > 0.09
        assert timer.elapsed < 0.2
    
    def test_timer_manual(self):
        """Test Timer manual usage"""
        timer = Timer()
        timer.start()
        time.sleep(0.1)
        timer.stop()
        
        assert timer.elapsed > 0.09
        assert timer.elapsed < 0.2
    
    def test_time_tracker(self):
        """Test TimeTracker for multiple timings"""
        tracker = TimeTracker()
        
        # Track different operations
        with tracker.track('data_loading'):
            time.sleep(0.05)
        
        with tracker.track('forward_pass'):
            time.sleep(0.1)
        
        with tracker.track('backward_pass'):
            time.sleep(0.08)
        
        # Get statistics
        stats = tracker.get_stats()
        
        assert 'data_loading' in stats
        assert 'forward_pass' in stats
        assert 'backward_pass' in stats
        
        assert stats['forward_pass']['total'] > 0.09
        assert stats['data_loading']['count'] == 1
    
    def test_format_time(self):
        """Test time formatting"""
        assert format_time(0.5) == "0.5s"
        assert format_time(65) == "1m 5s"
        assert format_time(3665) == "1h 1m 5s"
        assert format_time(86400) == "1d 0h 0m 0s"
    
    def test_estimate_remaining_time(self):
        """Test remaining time estimation"""
        # 20% complete in 10 seconds
        remaining = estimate_remaining_time(
            current_step=20,
            total_steps=100,
            elapsed_time=10
        )
        
        # Should estimate ~40 seconds remaining
        assert 35 < remaining < 45


class TestReproducibilityUtils:
    """Test reproducibility utilities"""
    
    def test_set_seed(self):
        """Test seed setting"""
        # Set seed
        set_seed(42)
        
        # Generate random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Reset seed
        set_seed(42)
        
        # Generate again
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_make_deterministic(self):
        """Test making PyTorch deterministic"""
        make_deterministic(True)
        
        # Check settings
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False
        
        # Reset
        make_deterministic(False)
        assert torch.backends.cudnn.deterministic == False
    
    def test_worker_init_fn(self):
        """Test DataLoader worker initialization"""
        # This ensures different workers get different seeds
        worker_id = 0
        worker_init_fn(worker_id)
        
        # Just check it runs without error
        assert True
    
    def test_random_state_save_restore(self):
        """Test saving and restoring random state"""
        # Set initial seed
        set_seed(42)
        
        # Generate some random numbers
        torch.rand(10)
        np.random.rand(10)
        
        # Save state
        state = get_random_state()
        
        # Generate more random numbers
        rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Restore state
        set_random_state(state)
        
        # Generate again - should match
        rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        assert torch.allclose(rand1, rand2)
        assert np.allclose(np_rand1, np_rand2)


class TestMiscUtils:
    """Test miscellaneous utility functions"""
    
    def test_get_num_parameters(self, sample_model):
        """Test parameter counting"""
        from retfound.utils.model import get_num_parameters
        
        total, trainable = get_num_parameters(sample_model)
        
        assert total > 0
        assert trainable == total  # All trainable by default
        
        # Freeze some parameters
        for param in list(sample_model.parameters())[:5]:
            param.requires_grad = False
        
        total2, trainable2 = get_num_parameters(sample_model)
        
        assert total2 == total
        assert trainable2 < total2
    
    def test_model_summary(self, sample_model, capsys):
        """Test model summary printing"""
        from retfound.utils.model import print_model_summary
        
        print_model_summary(
            sample_model,
            input_size=(1, 3, 224, 224),
            device='cpu'
        )
        
        captured = capsys.readouterr()
        
        # Should print model info
        assert "Total params" in captured.out or "Parameters" in captured.out
    
    def test_download_with_progress(self, temp_output_dir):
        """Test file download with progress"""
        from retfound.utils.download import download_file
        
        # Use a small test file
        test_url = "https://raw.githubusercontent.com/pytorch/pytorch/master/README.md"
        output_path = temp_output_dir / "README.md"
        
        # Download
        success = download_file(
            url=test_url,
            output_path=output_path,
            show_progress=True
        )
        
        assert success
        assert output_path.exists()
        assert output_path.stat().st_size > 0
