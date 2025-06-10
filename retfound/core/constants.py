"""Global constants for RETFound Training Framework."""

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

# Default CAASI class mapping
CAASI_CLASS_MAPPING = {
    # Diabetic Retinopathy stages
    'normal': 0, '0_normal': 0, 'normal_fundus': 0,
    'mild': 1, '1_mild': 1, 'dr_mild': 1, '1_dr_mild': 1,
    'moderate': 2, '2_moderate': 2, 'dr_moderate': 2, '2_dr_moderate': 2,
    'severe': 3, '3_severe': 3, 'dr_severe': 3, '3_dr_severe': 3,
    'proliferative': 4, '4_proliferative': 4, 'dr_proliferative': 4, '4_dr_proliferative': 4,
    
    # OCT pathologies
    'normal_oct': 5, '5_normal_oct': 5,
    'amd': 6, '6_amd': 6, 'age_related_macular_degeneration': 6,
    'dme': 7, '7_dme': 7, 'diabetic_macular_edema': 7, 'dme_fundus': 7,
    'erm': 8, '8_erm': 8, 'epiretinal_membrane': 8,
    'rao': 9, '9_rao': 9, 'retinal_artery_occlusion': 9,
    'rvo': 10, '10_rvo': 10, 'retinal_vein_occlusion': 10,
    'vid': 11, '11_vid': 11, 'vitreomacular_interface_disease': 11,
    'cnv': 12, '12_cnv': 12, 'choroidal_neovascularization': 12,
    'dme_oct': 13, '13_dme_oct': 13,
    'drusen': 14, '14_drusen': 14,
    
    # Glaucoma
    'glaucoma_normal': 15, '15_glaucoma_normal': 15,
    'glaucoma_positive': 16, '16_glaucoma_positive': 16,
    'glaucoma_suspect': 17, '17_glaucoma_suspect': 17,
    
    # Additional conditions
    'myopia': 18, '18_myopia': 18,
    'cataract': 19, '19_cataract': 19,
    'retinal_detachment': 20, '20_retinal_detachment': 20,
    'other': 21, '21_other': 21
}

# Proper class names for display
CLASS_NAMES = {
    0: 'Normal_Fundus',
    1: 'DR_Mild',
    2: 'DR_Moderate',
    3: 'DR_Severe',
    4: 'DR_Proliferative',
    5: 'Normal_OCT',
    6: 'AMD',
    7: 'DME_Fundus',
    8: 'ERM',
    9: 'RAO',
    10: 'RVO',
    11: 'VID',
    12: 'CNV',
    13: 'DME_OCT',
    14: 'DRUSEN',
    15: 'Glaucoma_Normal',
    16: 'Glaucoma_Positive',
    17: 'Glaucoma_Suspect',
    18: 'Myopia',
    19: 'Cataract',
    20: 'Retinal_Detachment',
    21: 'Other'
}

# Critical conditions requiring high sensitivity
CRITICAL_CONDITIONS = {
    'RAO': {
        'min_sensitivity': 0.99,
        'reason': 'Retinal artery occlusion - Emergency'
    },
    'RVO': {
        'min_sensitivity': 0.97,
        'reason': 'Retinal vein occlusion - Urgent'
    },
    'Retinal_Detachment': {
        'min_sensitivity': 0.99,
        'reason': 'Surgical emergency'
    },
    'CNV': {
        'min_sensitivity': 0.98,
        'reason': 'Risk of rapid vision loss'
    },
    'DR_Proliferative': {
        'min_sensitivity': 0.98,
        'reason': 'Risk of vitreous hemorrhage'
    },
    'DME': {
        'min_sensitivity': 0.95,
        'reason': 'Leading cause of vision loss in diabetics'
    },
    'Glaucoma_Positive': {
        'min_sensitivity': 0.95,
        'reason': 'Irreversible vision loss'
    }
}

# Image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RETFOUND_MEAN = [0.5, 0.5, 0.5]
RETFOUND_STD = [0.5, 0.5, 0.5]

# Training defaults
DEFAULT_NUM_WORKERS = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 100
DEFAULT_WARMUP_EPOCHS = 10

# Hardware constraints
MIN_GPU_MEMORY_GB = 16
RECOMMENDED_GPU_MEMORY_GB = 40

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
║                   RETFound Training Framework                  ║
║                        Version 2.0.0                          ║
║                    CAASI Medical AI Team                      ║
╚═══════════════════════════════════════════════════════════════╝
"""