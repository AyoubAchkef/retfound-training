#!/usr/bin/env python3
"""
Script pour corriger automatiquement tous les accès directs aux attributs de configuration
dans les fichiers de training de RETFound.

Ce script remplace tous les accès directs comme config.attribute par getattr(config, 'attribute', default_value)
"""

import os
import re
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mapping des attributs de configuration avec leurs valeurs par défaut
CONFIG_DEFAULTS = {
    # Training parameters
    'epochs': 50,
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
    'log_interval': 10,
    'val_frequency': 1,
    'save_frequency': 5,
    
    # Optimization parameters
    'base_lr': 1e-4,
    'min_lr': 1e-6,
    'warmup_lr': 1e-6,
    'warmup_epochs': 0,
    'weight_decay': 0.01,
    'layer_decay': 0.75,
    'gradient_clip': 1.0,
    'gradient_accumulation_steps': 1,
    'use_amp': True,
    'amp_dtype': 'float16',
    'use_compile': False,
    'compile_mode': 'default',
    'use_ema': False,
    'ema_decay': 0.9999,
    'ema_update_after_step': 100,
    'ema_update_every': 10,
    
    # Optimizer parameters
    'optimizer': 'adamw',
    'use_sam': False,
    'sam_rho': 0.05,
    'sam_adaptive': False,
    'adam_betas': (0.9, 0.999),
    'adam_epsilon': 1e-8,
    
    # Loss parameters
    'use_class_weights': True,
    'use_focal_loss': False,
    'focal_gamma': 2.0,
    'label_smoothing': 0.0,
    'num_classes': 28,
    
    # Augmentation parameters
    'use_mixup': False,
    'use_cutmix': False,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'mixup_prob': 0.5,
    'cutmix_prob': 0.5,
    
    # Early stopping parameters
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 1e-4,
    
    # Paths
    'output_path': 'outputs',
    'checkpoint_path': 'checkpoints',
    
    # Monitoring
    'use_tensorboard': False,
    'use_wandb': False,
    'wandb_project': 'retfound',
    'wandb_entity': None,
    
    # Distributed training
    'rank': 0,
    'local_rank': 0,
    'world_size': 1,
    'gpu_id': 0,
    'dist_backend': 'nccl',
    'find_unused_parameters': False,
    
    # Model parameters
    'input_size': 224,
    'modality': 'both',
}

def get_default_value_str(attr_name):
    """Retourne la valeur par défaut sous forme de string pour un attribut"""
    default = CONFIG_DEFAULTS.get(attr_name)
    if default is None:
        return 'None'
    elif isinstance(default, str):
        return f"'{default}'"
    elif isinstance(default, tuple):
        return str(default)
    else:
        return str(default)

def fix_config_access_in_file(file_path):
    """Corrige tous les accès directs aux attributs de configuration dans un fichier"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Pattern pour détecter config.attribute
        pattern = r'\bconfig\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        def replace_config_access(match):
            attr_name = match.group(1)
            
            # Ignorer certains cas spéciaux
            if attr_name in ['to_dict', 'data', 'model', 'training', 'optimization', 'augmentation']:
                return match.group(0)  # Ne pas remplacer
            
            default_value = get_default_value_str(attr_name)
            replacement = f"getattr(config, '{attr_name}', {default_value})"
            changes_made.append(f"  {attr_name} -> {replacement}")
            return replacement
        
        # Remplacer tous les accès directs
        content = re.sub(pattern, replace_config_access, content)
        
        # Sauvegarder seulement si des changements ont été faits
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Corrigé {file_path}")
            for change in changes_made:
                logger.info(change)
            return True
        else:
            logger.info(f"⏭️  Aucun changement nécessaire dans {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement de {file_path}: {e}")
        return False

def main():
    """Fonction principale"""
    logger.info("🔧 Début de la correction automatique des accès aux attributs de configuration")
    
    # Répertoires à traiter
    training_dir = Path("retfound/training")
    
    if not training_dir.exists():
        logger.error(f"❌ Répertoire {training_dir} non trouvé")
        return
    
    # Trouver tous les fichiers Python
    python_files = list(training_dir.rglob("*.py"))
    
    logger.info(f"📁 Trouvé {len(python_files)} fichiers Python dans {training_dir}")
    
    fixed_files = 0
    total_files = 0
    
    for file_path in python_files:
        total_files += 1
        if fix_config_access_in_file(file_path):
            fixed_files += 1
    
    logger.info(f"\n🎉 Correction terminée:")
    logger.info(f"   📊 {fixed_files}/{total_files} fichiers modifiés")
    logger.info(f"   ✅ Tous les accès directs aux attributs de configuration ont été sécurisés")
    
    if fixed_files > 0:
        logger.info(f"\n📝 Pour committer les changements:")
        logger.info(f"   git add retfound/training/")
        logger.info(f"   git commit -m 'Fix: Automated safe config attribute access across all training modules'")

if __name__ == "__main__":
    main()
