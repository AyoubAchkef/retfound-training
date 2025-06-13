"""
Train Command - Dataset v6.1
============================

CLI command for training RETFound models on Dataset v6.1 (28 classes).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader
import json
import numpy as np

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from retfound.core.config import RETFoundConfig
from retfound.core.constants import (
    NUM_TOTAL_CLASSES, UNIFIED_CLASS_NAMES, 
    CRITICAL_CONDITIONS, DATASET_V61_STATS
)
from retfound.models.factory import create_model
from retfound.data.datamodule import RETFoundDataModule
from retfound.training.trainer import RETFoundTrainer
from retfound.training.callbacks import (
    CheckpointCallback, EarlyStoppingCallback, MetricsCallback,
    VisualizationCallback, WandbCallback, TensorBoardCallback
)
from retfound.training.callbacks.metrics import ClassWiseMetricsCallback, create_standard_metrics
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
    parser.add_argument('--modality', type=str, default='both',
                       choices=['fundus', 'oct', 'both'],
                       help='Modality to train on (dataset v6.1)')
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
    parser.add_argument('--monitor-critical', action='store_true', default=True,
                       help='Monitor critical conditions (RAO, RVO, etc.)')
    parser.add_argument('--unified-classes', action='store_true', default=True,
                       help='Use unified 28-class system')


def validate_environment():
    """Validate the training environment"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! RETFound requires GPU.")
    
    # Check GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    
    logger.info(f"GPU detected: {gpu_name} with {gpu_memory_gb:.1f}GB memory")
    
    # Memory recommendations for RETFound
    memory_recommendations = {
        'optimal': 40,  # A100 40GB
        'recommended': 24,  # RTX 3090/4090
        'minimum': 16  # V100 16GB
    }
    
    memory_optimizations = False
    if gpu_memory_gb < memory_recommendations['minimum']:
        logger.warning(f"GPU has only {gpu_memory_gb:.1f}GB memory. RETFound needs {memory_recommendations['minimum']}GB+")
        logger.warning("Enabling aggressive memory optimizations...")
        memory_optimizations = True
    elif gpu_memory_gb < memory_recommendations['recommended']:
        logger.info(f"GPU has {gpu_memory_gb:.1f}GB memory. Enabling memory optimizations...")
        memory_optimizations = True
    
    return {
        'memory_optimizations': memory_optimizations,
        'gpu_memory_gb': gpu_memory_gb,
        'gpu_name': gpu_name
    }


def log_dataset_info(config: RETFoundConfig):
    """Log dataset v6.1 information"""
    logger.info("\n" + "="*60)
    logger.info("DATASET V6.1 INFORMATION")
    logger.info("="*60)
    
    modality = config.data.modality if hasattr(config.data, 'modality') else 'both'
    logger.info(f"Modality: {modality}")
    
    if modality == 'fundus':
        logger.info(f"Total images: {DATASET_V61_STATS['fundus']['total']:,}")
        logger.info(f"Classes: 18 Fundus classes")
    elif modality == 'oct':
        logger.info(f"Total images: {DATASET_V61_STATS['oct']['total']:,}")
        logger.info(f"Classes: 10 OCT classes")
    else:  # both
        logger.info(f"Total images: {DATASET_V61_STATS['total']['images']:,}")
        logger.info(f"Classes: 28 unified classes (18 Fundus + 10 OCT)")
        logger.info(f"Distribution: {DATASET_V61_STATS['fundus']['percentage']:.1f}% Fundus, "
                   f"{DATASET_V61_STATS['oct']['percentage']:.1f}% OCT")
    
    # Log critical conditions
    if config.monitor_critical:
        logger.info("\nCritical conditions monitored:")
        for condition, info in CRITICAL_CONDITIONS.items():
            logger.info(f"  - {condition}: Min sensitivity {info['min_sensitivity']:.2f}")
    
    logger.info("="*60 + "\n")


def setup_callbacks(config: RETFoundConfig, trial_name: str) -> List:
    """Setup training callbacks for v6.1"""
    callbacks = []
    
    # Determine monitoring metric
    monitor_metric = 'val_auc_macro'  # Best for medical imaging with multiple classes
    
    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        checkpoint_dir=config.checkpoint_path / trial_name,
        save_frequency=config.training.save_frequency,
        save_best=True,
        monitor='val_accuracy',
        mode='max',
        verbose=True
    ))
    
    # Early stopping
    callbacks.append(EarlyStoppingCallback(
        patience=config.training.early_stopping_patience if hasattr(config.training, 'early_stopping_patience') else 20,
        min_delta=config.training.early_stopping_min_delta if hasattr(config.training, 'early_stopping_min_delta') else 0.001,
        monitor=monitor_metric,
        mode='max'
    ))
    
    # Metrics tracking with v6.1 class names
    num_classes = config.model.num_classes if hasattr(config.model, 'num_classes') else NUM_TOTAL_CLASSES
    
    # Standard metrics callback
    callbacks.append(MetricsCallback(
        metrics=create_standard_metrics(),
        compute_on_train=True,
        compute_on_val=True
    ))
    
    # Class-wise metrics callback
    callbacks.append(ClassWiseMetricsCallback(
        num_classes=num_classes,
        class_names=UNIFIED_CLASS_NAMES[:num_classes],
        metrics_to_track=['sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    ))
    
    # Visualization
    callbacks.append(VisualizationCallback(
        output_dir=config.output_path / 'plots',
        frequency=config.training.val_frequency if hasattr(config.training, 'val_frequency') else 1,
        plot_confusion_matrix=True,
        plot_roc_curves=True,
        plot_per_class_metrics=True
    ))
    
    # Logging callbacks
    if hasattr(config.monitoring, 'use_tensorboard') and config.monitoring.use_tensorboard:
        callbacks.append(TensorBoardCallback(
            log_dir=config.output_path / 'tensorboard' / trial_name
        ))
    
    if hasattr(config.monitoring, 'use_wandb') and config.monitoring.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        callbacks.append(WandbCallback(
            log_frequency=config.training.log_interval if hasattr(config.training, 'log_interval') else 10,
            log_gradients=True,
            log_weights=True
        ))
    
    return callbacks


def train_single_fold(config: RETFoundConfig, fold: Optional[int] = None) -> Dict[str, Any]:
    """Train a single model (regular or one fold) for v6.1"""
    # Set seed for reproducibility
    set_seed(config.seed if hasattr(config, 'seed') else 42)
    
    # Create trial name
    trial_name = config.name if hasattr(config, 'name') else f'retfound_v61_{config.data.modality}'
    if fold is not None:
        trial_name = f"{trial_name}_fold{fold}"
    
    # Log dataset info
    log_dataset_info(config)
    
    # Setup data module
    logger.info("Setting up data module...")
    data_module = RETFoundDataModule(config)
    data_module.setup(fold=fold)
    
    # Update config with actual number of classes
    if hasattr(config, 'model'):
        config.model.num_classes = data_module.num_classes
    else:
        config.num_classes = data_module.num_classes
    
    logger.info(f"Number of classes: {data_module.num_classes}")
    logger.info(f"Training samples: {len(data_module.train_dataset):,}")
    logger.info(f"Validation samples: {len(data_module.val_dataset):,}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model('retfound', config)
    
    # Load pretrained weights
    weights_key = config.weights_key if hasattr(config, 'weights_key') else 'cfp'
    
    # Choose weights based on modality for v6.1
    if hasattr(config.data, 'modality'):
        if config.data.modality == 'oct':
            weights_key = 'oct'  # Use OCT weights for OCT-only training
        elif config.data.modality == 'both':
            weights_key = 'cfp'  # CFP weights work well for mixed modality
    
    weights_path = config.weights_paths[weights_key]
    logger.info(f"Loading pretrained weights from {weights_path} (key: {weights_key})")
    
    try:
        loading_info = model.load_pretrained_weights(weights_path, model_key=weights_key)
        logger.info(f"Successfully loaded {len(loading_info['loaded_keys'])} pretrained layers")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        logger.warning("Training from scratch...")
    
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
        config=config
    )
    
    # Add callbacks to trainer
    for callback in callbacks:
        trainer.callback_handler.add_callback(callback)
    
    # Setup training components
    trainer.setup_training(data_module.train_dataset)
    
    # Resume if specified
    start_epoch = 0
    if hasattr(config, 'resume_path') and config.resume_path:
        logger.info(f"Resuming from checkpoint: {config.resume_path}")
        trainer.load_checkpoint(config.resume_path)
        start_epoch = trainer.epoch + 1
    
    # Train
    logger.info(f"Starting training from epoch {start_epoch}...")
    history = trainer.train(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        start_epoch=start_epoch
    )
    
    # Log final summary
    trainer.log_summary()
    
    # Get best metrics
    best_metrics = {
        'best_epoch': trainer.best_epoch,
        'best_metric': trainer.best_metric,
        'history': history
    }
    
    # Extract key metrics from history
    if 'val_accuracy' in history:
        best_metrics['best_val_acc'] = max(history['val_accuracy']) * 100
    if 'val_auc_macro' in history:
        best_metrics['best_val_auc'] = max(history['val_auc_macro'])
    
    # Add critical condition performance
    if config.monitor_critical:
        critical_performance = {}
        for condition in CRITICAL_CONDITIONS.keys():
            key = f'val_{condition}_acc'
            if key in history and history[key]:
                critical_performance[condition] = {
                    'final': history[key][-1],
                    'best': max(history[key]),
                    'target': CRITICAL_CONDITIONS[condition]['min_sensitivity']
                }
        best_metrics['critical_conditions'] = critical_performance
    
    if fold is not None:
        best_metrics['fold'] = fold
    
    # Save final metrics
    metrics_path = config.output_path / f'final_metrics_{trial_name}.json'
    with open(metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    logger.info(f"Final metrics saved to {metrics_path}")
    
    return best_metrics


def train_kfold(config: RETFoundConfig) -> Dict[str, Any]:
    """Train with k-fold cross-validation for v6.1"""
    n_folds = config.training.n_folds if hasattr(config.training, 'n_folds') else 5
    
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    logger.info(f"Dataset: v6.1 with {NUM_TOTAL_CLASSES} classes")
    
    fold_results = []
    
    for fold in range(n_folds):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'='*70}")
        
        # Create fold-specific config
        fold_config = config  # In practice, you'd deep copy this
        if hasattr(fold_config, 'output_path'):
            fold_config.output_path = config.output_path / f'fold_{fold+1}'
        if hasattr(fold_config, 'checkpoint_path'):
            fold_config.checkpoint_path = config.checkpoint_path / f'fold_{fold+1}'
        
        # Train fold
        results = train_single_fold(fold_config, fold=fold)
        fold_results.append(results)
        
        # Log fold results
        if 'best_val_acc' in results:
            logger.info(f"Fold {fold+1} completed - Best Acc: {results['best_val_acc']:.2f}%")
        if 'best_val_auc' in results:
            logger.info(f"Fold {fold+1} completed - Best AUC: {results['best_val_auc']:.4f}")
    
    # Aggregate results
    logger.info(f"\n{'='*70}")
    logger.info("K-FOLD CROSS-VALIDATION RESULTS - Dataset v6.1")
    logger.info(f"{'='*70}")
    
    # Calculate statistics
    accuracies = [r.get('best_val_acc', 0) for r in fold_results]
    aucs = [r.get('best_val_auc', 0) for r in fold_results]
    
    summary = {
        'dataset_version': '6.1',
        'num_classes': NUM_TOTAL_CLASSES,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs)
    }
    
    logger.info(f"Average Accuracy: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
    logger.info(f"Average AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    
    # Aggregate critical condition performance
    if config.monitor_critical:
        logger.info("\nCritical Conditions Performance (averaged):")
        for condition in CRITICAL_CONDITIONS.keys():
            condition_scores = []
            for result in fold_results:
                if 'critical_conditions' in result and condition in result['critical_conditions']:
                    condition_scores.append(result['critical_conditions'][condition]['best'])
            
            if condition_scores:
                mean_score = np.mean(condition_scores)
                std_score = np.std(condition_scores)
                target = CRITICAL_CONDITIONS[condition]['min_sensitivity']
                logger.info(f"  {condition}: {mean_score:.3f} ± {std_score:.3f} (target: {target:.3f})")
                
                summary[f'{condition}_mean'] = mean_score
                summary[f'{condition}_std'] = std_score
    
    # Save summary
    summary_path = config.output_path / 'kfold_summary_v61.json'
    with open(summary_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_summary = {}
        for k, v in summary.items():
            if isinstance(v, (np.floating, np.integer)):
                json_summary[k] = float(v)
            else:
                json_summary[k] = v
        json.dump(json_summary, f, indent=2)
    
    logger.info(f"\nK-fold summary saved to {summary_path}")
    
    return summary


def run_train(args) -> int:
    """Main training function for dataset v6.1"""
    try:
        # Setup logging
        setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
        
        logger.info("="*60)
        logger.info("RETFound Training - Dataset v6.1")
        logger.info("="*60)
        
        # Load or create configuration
        if args.config:
            config = RETFoundConfig.load(Path(args.config))
            logger.info(f"Configuration loaded from {args.config}")
        else:
            # Use default v6.1 config
            config = RETFoundConfig()
            logger.info("Using default configuration for dataset v6.1")
        
        # Override config with command line arguments
        if args.epochs is not None:
            if hasattr(config, 'training'):
                config.training.epochs = args.epochs
            else:
                config.epochs = args.epochs
                
        if args.batch_size is not None:
            if hasattr(config, 'training'):
                config.training.batch_size = args.batch_size
            else:
                config.batch_size = args.batch_size
                
        if args.lr is not None:
            if hasattr(config, 'training'):
                config.training.base_lr = args.lr
            else:
                config.base_lr = args.lr
                
        if args.no_wandb:
            if hasattr(config, 'monitoring'):
                config.monitoring.use_wandb = False
            else:
                config.use_wandb = False
                
        if args.resume:
            config.resume_path = Path(args.resume)
            
        if args.name:
            config.name = args.name
            
        # Dataset v6.1 specific settings
        if hasattr(config, 'data'):
            config.data.modality = args.modality
            config.data.unified_classes = args.unified_classes
        else:
            config.modality = args.modality
            config.unified_classes = args.unified_classes
            
        # Store additional args
        config.weights_key = args.weights
        config.seed = args.seed
        config.monitor_critical = args.monitor_critical
        
        # Debug mode modifications
        if args.debug:
            if hasattr(config, 'training'):
                config.training.epochs = 3
                config.training.log_interval = 1
                config.training.save_frequency = 1
                config.training.val_frequency = 1
            else:
                config.epochs = 3
                config.log_interval = 1
                config.save_frequency = 1
                config.val_frequency = 1
            logger.info("Debug mode enabled - reduced epochs and frequencies")
        
        # Validate environment
        env_info = validate_environment()
        if env_info['memory_optimizations']:
            if hasattr(config, 'optimization'):
                config.optimization.use_gradient_checkpointing = True
            else:
                config.use_gradient_checkpointing = True
                
            if hasattr(config, 'training'):
                config.training.batch_size = min(config.training.batch_size, 8)
            else:
                config.batch_size = min(config.batch_size, 8)
                
            logger.info("Memory optimizations enabled")
        
        # Setup W&B if enabled
        use_wandb = config.monitoring.use_wandb if hasattr(config, 'monitoring') else config.use_wandb
        if use_wandb and not args.no_wandb and WANDB_AVAILABLE:
            wandb_config = {
                'dataset': 'CAASI v6.1',
                'num_classes': NUM_TOTAL_CLASSES,
                'modality': args.modality,
                'gpu': env_info['gpu_name'],
                'gpu_memory': f"{env_info['gpu_memory_gb']:.1f}GB"
            }
            
            # Add config to wandb
            if hasattr(config, 'to_dict'):
                wandb_config.update(config.to_dict())
            
            wandb.init(
                project=config.monitoring.wandb_project if hasattr(config.monitoring, 'wandb_project') else 'caasi-retfound-v61',
                entity=config.monitoring.wandb_entity if hasattr(config.monitoring, 'wandb_entity') else None,
                config=wandb_config,
                name=args.name or f"retfound_v61_{args.modality}_{args.weights}",
                tags=args.tags or ["retfound", "v6.1", "medical", args.modality],
                resume="allow" if args.resume else False
            )
        elif use_wandb and not args.no_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B logging requested but wandb not available. Install with: pip install wandb")
        
        # Create output directories
        config.output_path.mkdir(parents=True, exist_ok=True)
        config.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_save_path = config.output_path / 'config_v61.yaml'
        config.save(config_save_path)
        logger.info(f"Configuration saved to {config_save_path}")
        
        # Run training
        if args.kfold:
            results = train_kfold(config)
        else:
            results = train_single_fold(config, fold=args.fold)
        
        # Print final summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY - Dataset v6.1")
        logger.info("="*70)
        
        if args.kfold:
            logger.info(f"Average performance across {results['n_folds']} folds:")
            logger.info(f"Accuracy: {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
            logger.info(f"AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        else:
            if 'best_val_acc' in results:
                logger.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
            if 'best_val_auc' in results:
                logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
            logger.info(f"Best epoch: {results['best_epoch']}")
        
        # Log critical conditions if monitored
        if config.monitor_critical and 'critical_conditions' in results:
            logger.info("\nCritical Conditions Final Performance:")
            for condition, perf in results['critical_conditions'].items():
                status = "✓" if perf['best'] >= perf['target'] else "✗"
                logger.info(f"  {status} {condition}: {perf['best']:.3f} (target: {perf['target']:.3f})")
        
        logger.info(f"\nAll outputs saved to: {config.output_path}")
        
        # Cleanup W&B
        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        
        # Cleanup W&B on error
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish(exit_code=1)
        
        return 1


def add_subparser(subparsers):
    """Add train subcommand to parser"""
    parser = subparsers.add_parser(
        'train',
        help='Train RETFound model on dataset v6.1',
        description='Train RETFound model with advanced features for medical imaging'
    )
    
    add_train_args(parser)
    parser.set_defaults(func=run_train)
    
    return parser
