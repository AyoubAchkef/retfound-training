"""
Global constants for RETFound Fine-Tuning Framework.
This framework is specifically designed for fine-tuning RETFound models
on the CAASI Dataset v6.1 for ophthalmology applications.
"""

from typing import Dict, List, Tuple

# Model architecture constants
VIT_CONFIGS = {
    "vit_base_patch16_224": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "vit_large_patch16_224": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "vit_huge_patch14_224": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
}

# Dataset v6.1 Statistics
DATASET_V61_STATS = {
    'fundus': {
        'total': 44815,
        'train': 35848,
        'val': 4472,
        'test': 4495,
        'percentage': 21.1
    },
    'oct': {
        'total': 167137,
        'train': 133813,
        'val': 16627,
        'test': 16697,
        'percentage': 78.9
    },
    'total': {
        'images': 211952,
        'train': 169661,
        'val': 21099,
        'test': 21192
    }
}

# Number of classes in v6.1
NUM_FUNDUS_CLASSES = 18
NUM_OCT_CLASSES = 10
NUM_TOTAL_CLASSES = 28

# Fundus class names (v6.1)
FUNDUS_CLASS_NAMES = [
    "00_Normal_Fundus",
    "01_DR_Mild",
    "02_DR_Moderate",
    "03_DR_Severe",
    "04_DR_Proliferative",
    "05_Glaucoma_Suspect",
    "06_Glaucoma_Positive",
    "07_RVO",
    "08_RAO",
    "09_Hypertensive_Retinopathy",
    "10_Drusen",
    "11_CNV_Wet_AMD",
    "12_Myopia_Degenerative",
    "13_Retinal_Detachment",
    "14_Macular_Scar",
    "15_Cataract_Suspected",
    "16_Optic_Disc_Anomaly",
    "17_Other"
]

# OCT class names (v6.1)
OCT_CLASS_NAMES = [
    "00_Normal_OCT",
    "01_DME",
    "02_CNV_OCT",
    "03_Dry_AMD",
    "04_ERM",
    "05_Vitreomacular_Interface_Disease",
    "06_CSR",
    "07_RVO_OCT",
    "08_Glaucoma_OCT",
    "09_RAO_OCT"
]

# Unified class names for combined training (28 classes)
UNIFIED_CLASS_NAMES = [
    # Fundus classes (0-17)
    "Fundus_Normal",
    "Fundus_DR_Mild",
    "Fundus_DR_Moderate",
    "Fundus_DR_Severe",
    "Fundus_DR_Proliferative",
    "Fundus_Glaucoma_Suspect",
    "Fundus_Glaucoma_Positive",
    "Fundus_RVO",
    "Fundus_RAO",
    "Fundus_Hypertensive_Retinopathy",
    "Fundus_Drusen",
    "Fundus_CNV_Wet_AMD",
    "Fundus_Myopia_Degenerative",
    "Fundus_Retinal_Detachment",
    "Fundus_Macular_Scar",
    "Fundus_Cataract_Suspected",
    "Fundus_Optic_Disc_Anomaly",
    "Fundus_Other",
    # OCT classes (18-27)
    "OCT_Normal",
    "OCT_DME",
    "OCT_CNV",
    "OCT_Dry_AMD",
    "OCT_ERM",
    "OCT_Vitreomacular_Interface_Disease",
    "OCT_CSR",
    "OCT_RVO",
    "OCT_Glaucoma",
    "OCT_RAO"
]

# Class mappings
FUNDUS_CLASS_TO_IDX = {name: idx for idx, name in enumerate(FUNDUS_CLASS_NAMES)}
OCT_CLASS_TO_IDX = {name: idx for idx, name in enumerate(OCT_CLASS_NAMES)}
UNIFIED_CLASS_TO_IDX = {name: idx for idx, name in enumerate(UNIFIED_CLASS_NAMES)}

# Proper class names for display (unified system)
CLASS_NAMES = {idx: name for idx, name in enumerate(UNIFIED_CLASS_NAMES)}

# Critical conditions requiring high sensitivity
CRITICAL_CONDITIONS = {
    'RAO': {
        'fundus_idx': 8,
        'oct_idx': 9,
        'unified_idx': [8, 27],
        'min_sensitivity': 0.99,
        'reason': 'Retinal artery occlusion - Emergency'
    },
    'RVO': {
        'fundus_idx': 7,
        'oct_idx': 7,
        'unified_idx': [7, 25],
        'min_sensitivity': 0.97,
        'reason': 'Retinal vein occlusion - Urgent'
    },
    'Retinal_Detachment': {
        'fundus_idx': 13,
        'oct_idx': None,
        'unified_idx': [13],
        'min_sensitivity': 0.99,
        'reason': 'Surgical emergency'
    },
    'CNV': {
        'fundus_idx': 11,
        'oct_idx': 2,
        'unified_idx': [11, 20],
        'min_sensitivity': 0.98,
        'reason': 'Risk of rapid vision loss'
    },
    'DR_Proliferative': {
        'fundus_idx': 4,
        'oct_idx': None,
        'unified_idx': [4],
        'min_sensitivity': 0.98,
        'reason': 'Risk of vitreous hemorrhage'
    },
    'DME': {
        'fundus_idx': None,
        'oct_idx': 1,
        'unified_idx': [19],
        'min_sensitivity': 0.95,
        'reason': 'Leading cause of vision loss in diabetics'
    },
    'Glaucoma_Positive': {
        'fundus_idx': 6,
        'oct_idx': 8,
        'unified_idx': [6, 26],
        'min_sensitivity': 0.95,
        'reason': 'Irreversible vision loss'
    }
}

# Class weights for handling minor imbalances (post-augmentation in v6.1)
CLASS_WEIGHTS_V61 = {
    # OCT classes with residual imbalance
    "04_ERM": 2.0,          # 0.4% - Minoritaire
    "07_RVO_OCT": 2.0,      # 0.4% - Minoritaire
    "09_RAO_OCT": 1.5,      # 0.5% - Minoritaire
    
    # Fundus classes
    "12_Myopia_Degenerative": 1.5,  # 1.3% - Légèrement sous-représentée
    
    # All other classes: 1.0 (well balanced in v6.1)
}

# Image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RETFOUND_MEAN = [0.5, 0.5, 0.5]
RETFOUND_STD = [0.5, 0.5, 0.5]

# Training defaults for RETFound
DEFAULT_NUM_WORKERS = 8
DEFAULT_BATCH_SIZE = 16        # For vit_large on A100 40GB
DEFAULT_LEARNING_RATE = 5e-5   # Optimal for RETFound fine-tuning
DEFAULT_EPOCHS = 100
DEFAULT_WARMUP_EPOCHS = 10
DEFAULT_LAYER_DECAY = 0.75     # Layer-wise LR decay for ViT

# Hardware constraints
MIN_GPU_MEMORY_GB = 16
RECOMMENDED_GPU_MEMORY_GB = 40

# Suggested batch sizes for RETFound models (for A100 40GB)
SUGGESTED_BATCH_SIZES = {
    "vit_base_patch16_224": 32,      # RETFound Base - if available
    "vit_large_patch16_224": 16,     # RETFound Large (standard)
    "vit_huge_patch14_224": 8,       # RETFound Huge - if available
    # Effective batch size with gradient accumulation
    "vit_large_patch16_224_effective": 64,  # 16 * 4 gradient accumulation
}

# Export formats
EXPORT_FORMATS = ['onnx', 'torchscript', 'tensorrt']

# Metric names
METRIC_NAMES = [
    'accuracy',
    'balanced_accuracy',
    'cohen_kappa',
    'matthews_correlation',
    'auc_macro',
    'auc_weighted',
    'mean_sensitivity',
    'mean_specificity',
    'dr_quadratic_kappa'
]

# Environment variable names
ENV_VARS = {
    'DATASET_PATH': 'DATASET_PATH',
    'OUTPUT_PATH': 'OUTPUT_PATH',
    'CHECKPOINT_PATH': 'CHECKPOINT_PATH',
    'CACHE_DIR': 'CACHE_DIR',
    'WANDB_API_KEY': 'WANDB_API_KEY',
    'WANDB_PROJECT': 'WANDB_PROJECT',
    'WANDB_ENTITY': 'WANDB_ENTITY',
    'CUDA_VISIBLE_DEVICES': 'CUDA_VISIBLE_DEVICES',
}

# Color codes for terminal output
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'reset': '\033[0m'
}

# ASCII art for banner
BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                RETFound Fine-Tuning Framework                 ║
║                        Version 2.0.0                          ║
║                    CAASI Medical AI Team                      ║
║               Dataset v6.1 - 211,952 images                   ║
╚═══════════════════════════════════════════════════════════════╝
"""

# Additional v6.1 specific constants
DATASET_VERSION = "6.1"
DATASET_RELEASE_DATE = "2025-06-10"

# Cache settings
ENABLE_CACHE = True
CACHE_SIZE_LIMIT_GB = 50

# Multi-GPU settings
DEFAULT_WORLD_SIZE = 1
DEFAULT_RANK = 0
DEFAULT_LOCAL_RANK = 0

# Logging
LOG_INTERVAL = 10
CHECKPOINT_INTERVAL = 5  # epochs
VALIDATION_INTERVAL = 1  # epochs

# Early stopping
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.001

# Augmentation probabilities
MIXUP_PROB = 0.5
CUTMIX_PROB = 0.5
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0

# Temperature scaling for calibration
DEFAULT_TEMPERATURE = 1.5
CALIBRATION_BINS = 15

# RETFound specific constants
RETFOUND_WEIGHTS = {
    "cfp": "RETFound_mae_natureCFP.pth",      # For fundus images
    "oct": "RETFound_mae_natureOCT.pth",      # For OCT images
    "meh": "RETFound_mae_meh.pth"             # Multi-ethnic Himalaya
}

# RETFound model variants
RETFOUND_MODELS = {
    "retfound_base": "vit_base_patch16_224",
    "retfound_large": "vit_large_patch16_224",  # Standard/Default
    "retfound_huge": "vit_huge_patch14_224"
}

# Default RETFound configuration
DEFAULT_RETFOUND_MODEL = "vit_large_patch16_224"
DEFAULT_RETFOUND_WEIGHTS = "cfp"  # Use CFP for fundus by default

# Alias pour compatibilité avec medical.py
DATASET_V61_CLASSES = UNIFIED_CLASS_NAMES
DATASET_V40_CLASSES = UNIFIED_CLASS_NAMES[:22]  # Pour rétrocompatibilité

# Mapping automatique des indices critiques
CRITICAL_CLASS_INDICES = {}
for condition, info in CRITICAL_CONDITIONS.items():
    indices = []
    for class_name in UNIFIED_CLASS_NAMES:
        if condition.lower() in class_name.lower():
            indices.append(UNIFIED_CLASS_NAMES.index(class_name))
    if indices:
        CRITICAL_CLASS_INDICES[condition] = indices
