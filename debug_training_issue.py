#!/usr/bin/env python3
"""
Script de diagnostic pour identifier pourquoi l'entraînement s'arrête
"""

import sys
import traceback
import logging
from pathlib import Path

# Setup logging pour capturer toutes les erreurs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_training.log')
    ]
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test tous les imports nécessaires"""
    logger.info("=== TEST DES IMPORTS ===")
    
    try:
        from retfound.core.config import RETFoundConfig
        logger.info("✅ RETFoundConfig import OK")
    except Exception as e:
        logger.error(f"❌ RETFoundConfig import failed: {e}")
        traceback.print_exc()
    
    try:
        from retfound.models.factory import create_model
        logger.info("✅ create_model import OK")
    except Exception as e:
        logger.error(f"❌ create_model import failed: {e}")
        traceback.print_exc()
    
    try:
        from retfound.data.datamodule import RETFoundDataModule
        logger.info("✅ RETFoundDataModule import OK")
    except Exception as e:
        logger.error(f"❌ RETFoundDataModule import failed: {e}")
        traceback.print_exc()
    
    try:
        from retfound.training.trainer import RETFoundTrainer
        logger.info("✅ RETFoundTrainer import OK")
    except Exception as e:
        logger.error(f"❌ RETFoundTrainer import failed: {e}")
        traceback.print_exc()

def test_config_loading():
    """Test le chargement de la configuration"""
    logger.info("=== TEST CHARGEMENT CONFIG ===")
    
    try:
        from retfound.core.config import RETFoundConfig
        config_path = Path("configs/runpod.yaml")
        
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return None
        
        logger.info(f"📁 Loading config from: {config_path}")
        config = RETFoundConfig.load(config_path)
        logger.info("✅ Config loaded successfully")
        
        # Log config details
        logger.info(f"Config type: {type(config)}")
        if hasattr(config, 'data'):
            logger.info(f"Data config: {config.data}")
        if hasattr(config, 'model'):
            logger.info(f"Model config: {config.model}")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_data_module(config):
    """Test la création du data module"""
    logger.info("=== TEST DATA MODULE ===")
    
    try:
        from retfound.data.datamodule import RETFoundDataModule
        
        logger.info("Creating data module...")
        data_module = RETFoundDataModule(config)
        logger.info("✅ Data module created")
        
        logger.info("Setting up data module...")
        data_module.setup()
        logger.info("✅ Data module setup complete")
        
        logger.info(f"Number of classes: {data_module.num_classes}")
        logger.info(f"Train dataset size: {len(data_module.train_dataset)}")
        logger.info(f"Val dataset size: {len(data_module.val_dataset)}")
        
        return data_module
        
    except Exception as e:
        logger.error(f"❌ Data module failed: {e}")
        traceback.print_exc()
        return None

def test_model_creation(config):
    """Test la création du modèle"""
    logger.info("=== TEST MODEL CREATION ===")
    
    try:
        from retfound.models.factory import create_model
        
        logger.info("Creating model...")
        model = create_model('retfound', config)
        logger.info("✅ Model created successfully")
        
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return None

def test_trainer_creation(model, config):
    """Test la création du trainer"""
    logger.info("=== TEST TRAINER CREATION ===")
    
    try:
        from retfound.training.trainer import RETFoundTrainer
        
        logger.info("Creating trainer...")
        trainer = RETFoundTrainer(model=model, config=config)
        logger.info("✅ Trainer created successfully")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Trainer creation failed: {e}")
        traceback.print_exc()
        return None

def test_full_pipeline():
    """Test le pipeline complet"""
    logger.info("=== TEST PIPELINE COMPLET ===")
    
    # Test imports
    test_imports()
    
    # Test config
    config = test_config_loading()
    if not config:
        return False
    
    # Test data module
    data_module = test_data_module(config)
    if not data_module:
        return False
    
    # Test model
    model = test_model_creation(config)
    if not model:
        return False
    
    # Test trainer
    trainer = test_trainer_creation(model, config)
    if not trainer:
        return False
    
    logger.info("✅ TOUS LES TESTS PASSÉS !")
    return True

def simulate_train_command():
    """Simule la commande train pour identifier le problème"""
    logger.info("=== SIMULATION COMMANDE TRAIN ===")
    
    try:
        # Import du module train
        from retfound.cli.commands.train import run_train
        
        # Création d'un objet args simulé
        class MockArgs:
            config = "configs/runpod.yaml"
            weights = "oct"
            modality = "oct"
            epochs = None
            batch_size = None
            lr = None
            resume = None
            kfold = False
            fold = None
            debug = True  # Mode debug pour réduire les epochs
            no_wandb = True  # Désactiver wandb pour le test
            seed = 42
            name = None
            tags = None
            monitor_critical = True
            unified_classes = True
        
        args = MockArgs()
        
        logger.info("Calling run_train with mock args...")
        result = run_train(args)
        logger.info(f"✅ run_train completed with result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ run_train failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🔍 DÉBUT DU DIAGNOSTIC TRAINING")
    logger.info("=" * 50)
    
    try:
        # Test pipeline complet
        success = test_full_pipeline()
        
        if success:
            logger.info("\n" + "=" * 50)
            logger.info("🧪 TEST SIMULATION COMMANDE TRAIN")
            simulate_train_command()
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        traceback.print_exc()
    
    logger.info("\n" + "=" * 50)
    logger.info("🔍 FIN DU DIAGNOSTIC")
    logger.info("📋 Vérifiez le fichier debug_training.log pour plus de détails")
