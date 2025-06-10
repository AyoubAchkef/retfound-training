"""
Training Integration Tests
==========================

Test the complete training pipeline end-to-end.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import time

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.data.datamodule import RETFoundDataModule
from retfound.training.trainer import RETFoundTrainer
from retfound.training.callbacks import (
    CheckpointCallback, EarlyStoppingCallback,
    MetricsCallback, VisualizationCallback
)
from retfound.utils.reproducibility import set_seed


@pytest.mark.integration
class TestTrainingPipeline:
    """Test complete training pipeline"""
    
    def test_minimal_training_run(self, test_data_dir, temp_output_dir, minimal_config):
        """Test minimal training run"""
        # Configure for minimal training
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.checkpoint_path = temp_output_dir / 'checkpoints'
        minimal_config.epochs = 2
        minimal_config.batch_size = 2
        minimal_config.num_workers = 0
        minimal_config.use_amp = False  # Disable for CPU
        minimal_config.use_wandb = False
        minimal_config.use_tensorboard = False
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create data module
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        # Create model
        model = create_model(minimal_config)
        
        # Create minimal callbacks
        callbacks = [
            MetricsCallback(num_classes=minimal_config.num_classes),
            CheckpointCallback(
                checkpoint_dir=minimal_config.checkpoint_path,
                save_frequency=1
            )
        ]
        
        # Create trainer
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=callbacks
        )
        
        # Setup training
        trainer.setup_training(data_module.train_dataset)
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            start_epoch=0
        )
        
        # Check training completed
        assert len(history['train_loss']) == minimal_config.epochs
        assert len(history['val_loss']) == minimal_config.epochs
        
        # Check metrics improved
        assert history['val_acc'][-1] > history['val_acc'][0]
        
        # Check checkpoint saved
        checkpoint_files = list(minimal_config.checkpoint_path.glob('*.pth'))
        assert len(checkpoint_files) > 0
    
    def test_training_with_early_stopping(self, test_data_dir, temp_output_dir, minimal_config):
        """Test training with early stopping"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 10  # Set high but expect early stopping
        minimal_config.batch_size = 2
        minimal_config.num_workers = 0
        
        # Create components
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        # Add early stopping callback
        callbacks = [
            MetricsCallback(num_classes=minimal_config.num_classes),
            EarlyStoppingCallback(
                patience=2,
                min_delta=10.0,  # High threshold to trigger early stopping
                monitor='val_loss',
                mode='min'
            )
        ]
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=callbacks
        )
        
        trainer.setup_training(data_module.train_dataset)
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Should stop early
        assert len(history['train_loss']) < minimal_config.epochs
    
    def test_training_with_visualization(self, test_data_dir, temp_output_dir, minimal_config):
        """Test training with visualization callback"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        # Add visualization callback
        vis_dir = temp_output_dir / 'visualizations'
        callbacks = [
            MetricsCallback(num_classes=minimal_config.num_classes),
            VisualizationCallback(
                output_dir=vis_dir,
                frequency=1
            )
        ]
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=callbacks
        )
        
        trainer.setup_training(data_module.train_dataset)
        trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Check visualizations created
        assert vis_dir.exists()
        plot_files = list(vis_dir.glob('*.png'))
        assert len(plot_files) > 0
    
    def test_resume_training(self, test_data_dir, temp_output_dir, minimal_config):
        """Test resuming training from checkpoint"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.checkpoint_path = temp_output_dir / 'checkpoints'
        minimal_config.epochs = 4
        minimal_config.save_frequency = 2
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        callbacks = [
            MetricsCallback(num_classes=minimal_config.num_classes),
            CheckpointCallback(
                checkpoint_dir=minimal_config.checkpoint_path,
                save_frequency=minimal_config.save_frequency
            )
        ]
        
        # First training run
        trainer1 = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=callbacks
        )
        
        trainer1.setup_training(data_module.train_dataset)
        history1 = trainer1.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            start_epoch=0
        )
        
        # Get checkpoint
        checkpoint_path = minimal_config.checkpoint_path / 'retfound_latest.pth'
        assert checkpoint_path.exists()
        
        # Resume training
        trainer2 = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=callbacks
        )
        
        trainer2.setup_training(data_module.train_dataset)
        
        # Load checkpoint
        checkpoint = trainer2.load_checkpoint(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        
        # Continue training
        history2 = trainer2.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            start_epoch=start_epoch
        )
        
        # Check continued from correct epoch
        assert len(history2['train_loss']) == minimal_config.epochs - start_epoch


@pytest.mark.integration
class TestOptimizationFeatures:
    """Test advanced optimization features"""
    
    def test_sam_optimizer(self, test_data_dir, temp_output_dir, minimal_config):
        """Test training with SAM optimizer"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        minimal_config.use_sam = True
        minimal_config.sam_rho = 0.05
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=[MetricsCallback(num_classes=minimal_config.num_classes)]
        )
        
        trainer.setup_training(data_module.train_dataset)
        
        # Check SAM optimizer created
        from retfound.training.optimizers import SAM
        assert isinstance(trainer.optimizer, SAM)
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Should complete without error
        assert len(history['train_loss']) == minimal_config.epochs
    
    def test_ema_training(self, test_data_dir, temp_output_dir, minimal_config):
        """Test training with EMA"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        minimal_config.use_ema = True
        minimal_config.ema_decay = 0.999
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=[MetricsCallback(num_classes=minimal_config.num_classes)]
        )
        
        trainer.setup_training(data_module.train_dataset)
        
        # Check EMA created
        assert trainer.ema is not None
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # EMA model should exist and be different from main model
        ema_params = list(trainer.ema.ema_model.parameters())
        main_params = list(trainer.model.parameters())
        
        # After training, parameters should be different
        different = False
        for ema_p, main_p in zip(ema_params, main_params):
            if not torch.allclose(ema_p, main_p):
                different = True
                break
        
        assert different
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Mixed precision requires CUDA")
    def test_mixed_precision_training(self, test_data_dir, temp_output_dir, minimal_config):
        """Test mixed precision training"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        minimal_config.use_amp = True
        minimal_config.amp_dtype = torch.float16
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        model = model.cuda()
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=[MetricsCallback(num_classes=minimal_config.num_classes)],
            device='cuda'
        )
        
        trainer.setup_training(data_module.train_dataset)
        
        # Check scaler created
        assert trainer.scaler is not None
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Should complete without error
        assert len(history['train_loss']) == minimal_config.epochs


@pytest.mark.integration
class TestDataAugmentation:
    """Test data augmentation in training"""
    
    def test_mixup_training(self, test_data_dir, temp_output_dir, minimal_config):
        """Test training with mixup"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        minimal_config.use_mixup = True
        minimal_config.mixup_alpha = 0.2
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=[MetricsCallback(num_classes=minimal_config.num_classes)]
        )
        
        trainer.setup_training(data_module.train_dataset)
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Should complete without error
        assert len(history['train_loss']) == minimal_config.epochs
    
    def test_cutmix_training(self, test_data_dir, temp_output_dir, minimal_config):
        """Test training with cutmix"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        minimal_config.use_cutmix = True
        minimal_config.cutmix_alpha = 1.0
        
        data_module = RETFoundDataModule(minimal_config)
        data_module.setup()
        
        model = create_model(minimal_config)
        
        trainer = RETFoundTrainer(
            model=model,
            config=minimal_config,
            callbacks=[MetricsCallback(num_classes=minimal_config.num_classes)]
        )
        
        trainer.setup_training(data_module.train_dataset)
        
        # Train
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Should complete without error
        assert len(history['train_loss']) == minimal_config.epochs


@pytest.mark.integration
@pytest.mark.slow
class TestKFoldTraining:
    """Test k-fold cross validation"""
    
    def test_kfold_training(self, test_data_dir, temp_output_dir, minimal_config):
        """Test k-fold cross validation training"""
        minimal_config.dataset_path = test_data_dir
        minimal_config.output_path = temp_output_dir
        minimal_config.epochs = 2
        minimal_config.use_kfold = True
        minimal_config.n_folds = 3
        
        all_results = []
        
        for fold in range(minimal_config.n_folds):
            # Create data module for fold
            data_module = RETFoundDataModule(minimal_config)
            data_module.setup(fold=fold)
            
            # Create new model for each fold
            model = create_model(minimal_config)
            
            # Create trainer
            trainer = RETFoundTrainer(
                model=model,
                config=minimal_config,
                callbacks=[MetricsCallback(num_classes=minimal_config.num_classes)]
            )
            
            trainer.setup_training(data_module.train_dataset)
            
            # Train fold
            history = trainer.train(
                train_loader=data_module.train_dataloader(),
                val_loader=data_module.val_dataloader()
            )
            
            # Collect results
            all_results.append({
                'fold': fold,
                'final_val_acc': history['val_acc'][-1],
                'final_val_loss': history['val_loss'][-1]
            })
        
        # Check all folds completed
        assert len(all_results) == minimal_config.n_folds
        
        # Calculate average performance
        avg_acc = np.mean([r['final_val_acc'] for r in all_results])
        assert avg_acc > 0


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios"""
    
    def test_medical_image_classification_pipeline(self, test_data_dir, temp_output_dir):
        """Test complete medical image classification pipeline"""
        # Create realistic configuration
        config = RETFoundConfig(
            dataset_path=test_data_dir,
            output_path=temp_output_dir,
            num_classes=3,
            epochs=3,
            batch_size=4,
            base_lr=1e-4,
            use_sam=True,
            use_ema=True,
            use_mixup=True,
            label_smoothing=0.1,
            use_amp=False,  # CPU testing
            num_workers=0
        )
        
        # Full pipeline
        results = run_training_pipeline(config)
        
        # Check results
        assert results['status'] == 'completed'
        assert results['best_val_acc'] > 0
        assert results['training_time'] > 0
        
        # Check outputs created
        assert (config.output_path / 'config.yaml').exists()
        assert (config.checkpoint_path / 'retfound_best.pth').exists()
        assert (config.output_path / 'training_report.json').exists()


def run_training_pipeline(config):
    """Helper to run complete training pipeline"""
    start_time = time.time()
    
    try:
        # Set seed
        set_seed(42)
        
        # Create data module
        data_module = RETFoundDataModule(config)
        data_module.setup()
        
        # Update config with detected classes
        config.num_classes = data_module.num_classes
        
        # Save config
        config.save(config.output_path / 'config.yaml')
        
        # Create model
        model = create_model(config)
        
        # Create callbacks
        callbacks = [
            MetricsCallback(
                num_classes=config.num_classes,
                class_names=data_module.class_names
            ),
            CheckpointCallback(
                checkpoint_dir=config.checkpoint_path,
                save_frequency=config.save_frequency,
                monitor='val_auc',
                mode='max'
            ),
            EarlyStoppingCallback(
                patience=5,
                min_delta=0.001,
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Create trainer
        trainer = RETFoundTrainer(
            model=model,
            config=config,
            callbacks=callbacks
        )
        
        # Setup and train
        trainer.setup_training(data_module.train_dataset)
        
        history = trainer.train(
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        # Save training report
        report = {
            'status': 'completed',
            'best_val_acc': trainer.best_val_acc,
            'best_val_auc': trainer.best_val_auc,
            'best_epoch': trainer.best_epoch,
            'total_epochs': len(history['train_loss']),
            'training_time': time.time() - start_time,
            'final_metrics': trainer.val_metrics.compute_metrics() if hasattr(trainer, 'val_metrics') else {}
        }
        
        with open(config.output_path / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'training_time': time.time() - start_time
        }
