"""Configuration management for RETFound Training Framework."""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import torch
from dataclasses_json import dataclass_json
import logging

from retfound.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    type: str = "vit_large_patch16_224"
    num_classes: int = 28
    patch_size: int = 16
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    pretrained_weights: Optional[str] = "cfp"  # cfp, oct, meh, or None
    
    def validate(self) -> None:
        """Validate model configuration."""
        if self.pretrained_weights and self.pretrained_weights not in ["cfp", "oct", "meh"]:
            raise ConfigurationError(
                f"Invalid pretrained_weights: {self.pretrained_weights}. "
                "Must be 'cfp', 'oct', 'meh', or None"
            )
        if self.num_classes < 1:
            raise ConfigurationError("num_classes must be >= 1")


@dataclass_json
@dataclass
class DataConfig:
    """Data configuration."""
    
    dataset_path: Path = Path("/workspace/CAASI-DATASET/01_CLASSIFICATION")
    input_size: int = 224
    pixel_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    pixel_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    augmentation_level: str = "medium"  # none, light, medium, strong
    use_pathology_augmentation: bool = True
    balance_dataset: bool = True
    
    def __post_init__(self):
        """Convert paths to Path objects."""
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if self.cache_dir and isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
    
    def validate(self) -> None:
        """Validate data configuration."""
        if not self.dataset_path.exists():
            raise ConfigurationError(f"Dataset path does not exist: {self.dataset_path}")
        if self.augmentation_level not in ["none", "light", "medium", "strong"]:
            raise ConfigurationError(
                f"Invalid augmentation_level: {self.augmentation_level}"
            )


@dataclass_json
@dataclass
class TrainingConfig:
    """Training configuration."""
    
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    epochs: int = 100
    val_frequency: int = 1
    save_frequency: int = 10
    log_interval: int = 10
    
    # Learning rate
    base_lr: float = 5e-5
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    warmup_lr: float = 1e-7
    layer_decay: float = 0.65
    
    # Loss
    label_smoothing: float = 0.1
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    
    # K-fold
    use_kfold: bool = False
    n_folds: int = 5
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ConfigurationError("epochs must be >= 1")
        if self.base_lr <= 0:
            raise ConfigurationError("base_lr must be > 0")


@dataclass_json
@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    
    optimizer: str = "adamw"
    adam_epsilon: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    
    # SAM
    use_sam: bool = True
    sam_rho: float = 0.05
    sam_adaptive: bool = True
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_after_step: int = 100
    ema_update_every: int = 10
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: Optional[str] = None  # "float16", "bfloat16", or None (auto)
    
    # Compile
    use_compile: bool = True
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    
    def validate(self) -> None:
        """Validate optimization configuration."""
        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ConfigurationError(f"Invalid optimizer: {self.optimizer}")
        if self.amp_dtype and self.amp_dtype not in ["float16", "bfloat16"]:
            raise ConfigurationError(f"Invalid amp_dtype: {self.amp_dtype}")


@dataclass_json
@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    
    use_mixup: bool = True
    mixup_alpha: float = 0.8
    mixup_prob: float = 0.5
    
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    
    # Test-time augmentation
    use_tta: bool = True
    tta_augmentations: int = 5
    
    # Temperature scaling
    use_temperature_scaling: bool = True
    calibration_bins: int = 15
    
    def validate(self) -> None:
        """Validate augmentation configuration."""
        if self.mixup_alpha < 0:
            raise ConfigurationError("mixup_alpha must be >= 0")
        if self.cutmix_alpha < 0:
            raise ConfigurationError("cutmix_alpha must be >= 0")


@dataclass_json
@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    
    use_tensorboard: bool = True
    use_wandb: bool = True
    wandb_project: str = "caasi-retfound"
    wandb_entity: Optional[str] = None
    
    # Metrics
    track_per_class_metrics: bool = True
    generate_confusion_matrix: bool = True
    generate_roc_curves: bool = True
    generate_clinical_report: bool = True
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        pass


@dataclass_json
@dataclass
class ExportConfig:
    """Export configuration."""
    
    export_onnx: bool = False
    export_torchscript: bool = True
    export_tensorrt: bool = False
    
    # Quantization
    quantize: bool = False
    quantization_backend: str = "qnnpack"  # "qnnpack", "fbgemm"
    
    # Optimization
    optimize_for_mobile: bool = False
    
    def validate(self) -> None:
        """Validate export configuration."""
        if self.quantization_backend not in ["qnnpack", "fbgemm"]:
            raise ConfigurationError(
                f"Invalid quantization_backend: {self.quantization_backend}"
            )


@dataclass_json
@dataclass
class RETFoundConfig:
    """Main configuration class for RETFound Training Framework."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Paths
    base_path: Path = Path("/workspace")
    output_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    weights_paths: Optional[Dict[str, Path]] = None
    
    # Hardware
    device: str = "cuda"
    cudnn_benchmark: bool = True
    
    # Random seed
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """Initialize derived attributes and paths."""
        # Convert sub-configs from dicts if needed
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.optimization, dict):
            self.optimization = OptimizationConfig(**self.optimization)
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationConfig(**self.augmentation)
        if isinstance(self.monitoring, dict):
            self.monitoring = MonitoringConfig(**self.monitoring)
        if isinstance(self.export, dict):
            self.export = ExportConfig(**self.export)
        
        # Set default paths
        if self.output_path is None:
            self.output_path = self.base_path / "outputs" / "retfound"
        if self.checkpoint_path is None:
            self.checkpoint_path = self.base_path / "checkpoints" / "retfound"
        if self.data.cache_dir is None:
            self.data.cache_dir = self.base_path / "caasi_cache" / "retfound"
        
        # Set default weights paths
        if self.weights_paths is None:
            self.weights_paths = {
                'cfp': self.base_path / "RETFound_mae_natureCFP.pth",
                'oct': self.base_path / "RETFound_mae_natureOCT.pth",
                'meh': self.base_path / "RETFound_mae_meh.pth"
            }
        
        # Convert paths
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)
        
        # Auto-detect AMP dtype
        if self.optimization.amp_dtype is None and self.optimization.use_amp:
            self.optimization.amp_dtype = (
                "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            )
        
        # Auto-detect compile availability
        if self.optimization.use_compile and not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, disabling")
            self.optimization.use_compile = False
    
    def validate(self) -> None:
        """Validate all configurations."""
        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.optimization.validate()
        self.augmentation.validate()
        self.monitoring.validate()
        self.export.validate()
        
        # Additional cross-configuration validation
        effective_batch_size = (
            self.training.batch_size * self.training.gradient_accumulation_steps
        )
        if effective_batch_size > 256:
            logger.warning(
                f"Very large effective batch size: {effective_batch_size}. "
                "This may require careful tuning of learning rate."
            )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle Path objects
        config_dict = asdict(self)
        config_dict = self._convert_paths_to_strings(config_dict)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "RETFoundConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "RETFoundConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if os.getenv("DATASET_PATH"):
            config.data.dataset_path = Path(os.getenv("DATASET_PATH"))
        if os.getenv("OUTPUT_PATH"):
            config.output_path = Path(os.getenv("OUTPUT_PATH"))
        if os.getenv("CHECKPOINT_PATH"):
            config.checkpoint_path = Path(os.getenv("CHECKPOINT_PATH"))
        
        # Training settings
        if os.getenv("DEFAULT_BATCH_SIZE"):
            config.training.batch_size = int(os.getenv("DEFAULT_BATCH_SIZE"))
        if os.getenv("DEFAULT_EPOCHS"):
            config.training.epochs = int(os.getenv("DEFAULT_EPOCHS"))
        if os.getenv("DEFAULT_LEARNING_RATE"):
            config.training.base_lr = float(os.getenv("DEFAULT_LEARNING_RATE"))
        
        # Optimization settings
        if os.getenv("USE_SAM_OPTIMIZER"):
            config.optimization.use_sam = os.getenv("USE_SAM_OPTIMIZER").lower() == "true"
        if os.getenv("USE_EMA"):
            config.optimization.use_ema = os.getenv("USE_EMA").lower() == "true"
        if os.getenv("USE_MIXED_PRECISION"):
            config.optimization.use_amp = os.getenv("USE_MIXED_PRECISION").lower() == "true"
        
        # Monitoring
        if os.getenv("USE_WANDB"):
            config.monitoring.use_wandb = os.getenv("USE_WANDB").lower() == "true"
        if os.getenv("WANDB_PROJECT"):
            config.monitoring.wandb_project = os.getenv("WANDB_PROJECT")
        if os.getenv("WANDB_ENTITY"):
            config.monitoring.wandb_entity = os.getenv("WANDB_ENTITY")
        
        return config
    
    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Recursively convert Path objects to strings for YAML serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_strings(v) for v in obj]
        return obj
    
    def get_device(self) -> torch.device:
        """Get the torch device based on configuration."""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(self.device)
    
    def get_amp_dtype(self) -> Optional[torch.dtype]:
        """Get the AMP dtype as torch.dtype."""
        if not self.optimization.use_amp:
            return None
        if self.optimization.amp_dtype == "bfloat16":
            return torch.bfloat16
        elif self.optimization.amp_dtype == "float16":
            return torch.float16
        return None