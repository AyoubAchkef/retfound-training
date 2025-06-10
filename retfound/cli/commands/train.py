"""
Train Command
=============

CLI command for training RETFound models.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
import wandb

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.data.datamodule import RETFoundDataModule
from retfound.training.trainer import RETFoundTrainer
from retfound.training.callbacks import (
    CheckpointCallback, EarlyStoppingCallback, MetricsCallback,
    VisualizationCallback, WandbCallback, TensorBoardCallback
)
from retfound.utils.logging import setup_logging
from retfound.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def add_train_args(parser):
    """Add training-specific arguments to parser"""
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--weights', type=str, default='cfp',
                       choices=['cfp', 'oct', 'meh'],
                       help='Which RETFound weights to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--kfold', action='store_true',
                       help='Use K-fold cross-validation')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train (for k-fold)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced epochs')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--tags', nargs='+', default=None,
                       help='Tags for the experiment')


def validate_environment():
    """Validate the training environment"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! RETFound requires GPU.")
    
    # Check GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb < 16:
        logger.warning(f"GPU has only {gpu_memory_gb:.1f}GB memory. RETFound needs 16GB+")
        logger.warning("Enabling memory optimizations...")
        return {'memory_optimizations': True}
    
    return {'memory_optimizations': False}


def setup_callbacks(config: RETFoundConfig, trial_name: str) -> list:
    """Setup training callbacks"""
    callbacks = []
    
    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        checkpoint_dir=config.checkpoint_path,
        save_frequency=config.save_frequency,
        monitor='val_auc',
        mode='max'
    ))
    
    # Early stopping
    callbacks.append(EarlyStoppingCallback(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        monitor='val_auc',
        mode='max'
    ))
    
    # Metrics tracking
    callbacks.append(MetricsCallback(
        num_classes=config.num_classes,
        class_names=None  # Will be set from dataset
    ))
    
    # Visualization
    callbacks.append(VisualizationCallback(
        output_dir=config.output_path / 'plots',
        frequency=10
    ))
    
    # Logging callbacks
    if config.use_tensorboard:
        callbacks.append(TensorBoardCallback(
            log_dir=config.output_path / 'tensorboard' / trial_name
        ))
    
    if config.use_wandb and wandb.run is not None:
        callbacks.append(WandbCallback())
    
    return callbacks


def train_single_fold(config: RETFoundConfig, fold: Optional[int] = None) -> Dict[str, Any]:
    """Train a single model (regular or one fold)"""
    # Set seed for reproducibility
    set_seed(config.seed if hasattr(config, 'seed') else 42)
    
    # Create trial name
    trial_name = config.name if hasattr(config, 'name') else 'retfound'
    if fold is not None:
        trial_name = f"{trial_name}_fold{fold}"
    
    # Setup data module
    logger.info("Setting up data module...")
    data_module = RETFoundDataModule(config)
    data_module.setup(fold=fold)
    
    # Update config with actual number of classes
    config.num_classes = data_module.num_classes
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Load pretrained weights
    weights_key = getattr(config, 'weights_key', 'cfp')
    weights_path = config.weights_paths[weights_key]
    logger.info(f"Loading pretrained weights from {weights_path}")
    model.load_pretrained_weights(weights_path, model_key=weights_key)
    
    # Setup callbacks
    callbacks = setup_callbacks(config, trial_name)
    
    # Update callbacks with class names
    for callback in callbacks:
        if hasattr(callback, 'class_names'):
            callback.class_names = data_module.class_names
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = RETFoundTrainer(
        model=model,
        config=config,
        callbacks=callbacks
    )
    
    # Setup training components
    trainer.setup_training(data_module.train_dataset)
    
    # Resume if specified
    start_epoch = 0
    if hasattr(config, 'resume_path') and config.resume_path:
        logger.info(f"Resuming from checkpoint: {config.resume_path}")
        checkpoint = trainer.load_checkpoint(config.resume_path)
        start_epoch = checkpoint['epoch'] + 1
    
    # Train
    logger.info(f"Starting training from epoch {start_epoch}...")
    history = trainer.train(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        start_epoch=start_epoch
    )
    
    # Get best metrics
    best_metrics = {
        'best_epoch': trainer.best_epoch,
        'best_val_acc': trainer.best_val_acc,
        'best_val_auc': trainer.best_val_auc,
        'history': history
    }
    
    if fold is not None:
        best_metrics['fold'] = fold
    
    return best_metrics


def train_kfold(config: RETFoundConfig) -> Dict[str, Any]:
    """Train with k-fold cross-validation"""
    import numpy as np
    
    logger.info(f"Starting {config.n_folds}-fold cross-validation...")
    
    fold_results = []
    
    for fold in range(config.n_folds):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold + 1}/{config.n_folds}")
        logger.info(f"{'='*70}")
        
        # Train fold
        fold_config = config.copy()
        fold_config.output_path = config.output_path / f'fold_{fold+1}'
        fold_config.checkpoint_path = config.checkpoint_path / f'fold_{fold+1}'
        
        results = train_single_fold(fold_config, fold=fold)
        fold_results.append(results)
        
        logger.info(f"Fold {fold+1} completed - Best Acc: {results['best_val_acc']:.2f}%, "
                   f"Best AUC: {results['best_val_auc']:.4f}")
    
    # Aggregate results
    logger.info(f"\n{'='*70}")
    logger.info("K-FOLD CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*70}")
    
    accuracies = [r['best_val_acc'] for r in fold_results]
    aucs = [r['best_val_auc'] for r in fold_results]
    
    summary = {
        'n_folds': config.n_folds,
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs)
    }
    
    logger.info(f"Average Accuracy: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
    logger.info(f"Average AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    
    # Save summary
    import json
    summary_path = config.output_path / 'kfold_summary.json'
    with open(summary_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_summary = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in summary.items()
        }
        json.dump(json_summary, f, indent=2)
    
    logger.info(f"\nK-fold summary saved to {summary_path}")
    
    return summary


def run_train(args) -> int:
    """Main training function"""
    try:
        # Setup logging
        setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
        
        # Load or create configuration
        if args.config:
            config = RETFoundConfig.load(Path(args.config))
            logger.info(f"Configuration loaded from {args.config}")
        else:
            config = RETFoundConfig()
        
        # Override config with command line arguments
        if args.epochs is not None:
            config.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.base_lr = args.lr
        if args.no_wandb:
            config.use_wandb = False
        if args.resume:
            config.resume_path = Path(args.resume)
        if args.name:
            config.name = args.name
        
        # Store additional args in config
        config.weights_key = args.weights
        config.seed = args.seed
        
        # Debug mode modifications
        if args.debug:
            config.epochs = 3
            config.log_interval = 1
            config.save_frequency = 1
            config.val_frequency = 1
            logger.info("Debug mode enabled - reduced epochs and frequencies")
        
        # Validate environment
        env_info = validate_environment()
        if env_info['memory_optimizations']:
            config.use_gradient_checkpointing = True
            config.batch_size = min(config.batch_size, 8)
        
        # Setup W&B if enabled
        if config.use_wandb and not args.no_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=args.name or f"retfound_{args.weights}",
                tags=args.tags or ["retfound", "medical", "ophthalmology"],
                resume="allow" if args.resume else False
            )
        
        # Create output directories
        config.output_path.mkdir(parents=True, exist_ok=True)
        config.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_save_path = config.output_path / 'config.yaml'
        config.save(config_save_path)
        logger.info(f"Configuration saved to {config_save_path}")
        
        # Run training
        if args.kfold:
            results = train_kfold(config)
        else:
            results = train_single_fold(config, fold=args.fold)
        
        # Print final summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        if args.kfold:
            logger.info(f"Average performance across {config.n_folds} folds:")
            logger.info(f"Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
            logger.info(f"AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        else:
            logger.info(f"Best epoch: {results['best_epoch']}")
            logger.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
            logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
        
        logger.info(f"\nAll outputs saved to: {config.output_path}")
        
        # Cleanup W&B
        if config.use_wandb and wandb.run is not None:
            wandb.finish()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        
        # Cleanup W&B on error
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        
        return 1
