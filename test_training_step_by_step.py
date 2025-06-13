#!/usr/bin/env python3
"""
Test l'entraînement étape par étape pour identifier où ça bloque
"""

import sys
import traceback
import logging
from pathlib import Path

# Setup logging détaillé
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_step_by_step.log')
    ]
)

logger = logging.getLogger(__name__)

def test_step_by_step():
    """Test chaque étape de l'entraînement"""
    
    logger.info("🔍 DÉBUT TEST ÉTAPE PAR ÉTAPE")
    
    try:
        # Étape 1: Imports
        logger.info("📦 Étape 1: Imports...")
        from retfound.core.config import RETFoundConfig
        from retfound.models.factory import create_model
        from retfound.data.datamodule import RETFoundDataModule
        from retfound.training.trainer import RETFoundTrainer
        logger.info("✅ Imports OK")
        
        # Étape 2: Configuration
        logger.info("⚙️ Étape 2: Chargement configuration...")
        config = RETFoundConfig.load(Path("configs/runpod.yaml"))
        logger.info("✅ Configuration OK")
        
        # Étape 3: Data Module
        logger.info("📊 Étape 3: Création data module...")
        data_module = RETFoundDataModule(config)
        logger.info("✅ Data module créé")
        
        logger.info("📊 Étape 3b: Setup data module...")
        data_module.setup()
        logger.info("✅ Data module setup OK")
        
        # Étape 4: Modèle
        logger.info("🤖 Étape 4: Création modèle...")
        model = create_model('retfound', config)
        logger.info("✅ Modèle créé")
        
        # Étape 5: Trainer
        logger.info("🏋️ Étape 5: Création trainer...")
        trainer = RETFoundTrainer(model=model, config=config)
        logger.info("✅ Trainer créé")
        
        # Étape 6: Setup training
        logger.info("🔧 Étape 6: Setup training...")
        trainer.setup_training(data_module.train_dataset)
        logger.info("✅ Training setup OK")
        
        # Étape 7: Test des dataloaders
        logger.info("📋 Étape 7: Test dataloaders...")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        logger.info(f"✅ Train loader: {len(train_loader)} batches")
        logger.info(f"✅ Val loader: {len(val_loader)} batches")
        
        # Étape 8: Test d'un batch
        logger.info("🧪 Étape 8: Test d'un batch...")
        for batch in train_loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, metadata = batch
            logger.info(f"✅ Batch shape: {images.shape}, Labels: {labels.shape}")
            break
        
        # Étape 9: Test forward pass
        logger.info("➡️ Étape 9: Test forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(images[:2])  # Test avec 2 échantillons
            logger.info(f"✅ Forward pass OK: {outputs.shape}")
        
        logger.info("🎉 TOUS LES TESTS PASSÉS !")
        return True
        
    except Exception as e:
        logger.error(f"❌ ERREUR à l'étape: {e}")
        logger.error("Traceback complet:")
        traceback.print_exc()
        return False

def simulate_train_command_minimal():
    """Simule la commande train de façon minimale"""
    logger.info("🚀 SIMULATION COMMANDE TRAIN MINIMALE")
    
    try:
        # Import direct de la fonction
        from retfound.cli.commands.train import run_train
        
        # Args minimaux
        class MinimalArgs:
            config = "configs/runpod.yaml"
            weights = "oct"
            modality = "oct"
            epochs = 1  # Une seule époque pour test
            batch_size = 2  # Batch size minimal
            lr = None
            resume = None
            kfold = False
            fold = None
            debug = True
            no_wandb = True  # Pas de wandb pour éviter les complications
            seed = 42
            name = "test_minimal"
            tags = None
            monitor_critical = True
            unified_classes = True
        
        args = MinimalArgs()
        
        logger.info("📞 Appel run_train...")
        result = run_train(args)
        logger.info(f"✅ run_train terminé avec résultat: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ run_train échoué: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🔍 DÉBUT DES TESTS DÉTAILLÉS")
    logger.info("=" * 60)
    
    # Import torch ici pour éviter les problèmes
    import torch
    
    # Test étape par étape
    success1 = test_step_by_step()
    
    if success1:
        logger.info("\n" + "=" * 60)
        logger.info("🧪 TEST SIMULATION COMMANDE TRAIN")
        success2 = simulate_train_command_minimal()
    
    logger.info("\n" + "=" * 60)
    logger.info("🔍 FIN DES TESTS")
    
    if success1:
        logger.info("✅ Tests étape par étape: SUCCÈS")
    else:
        logger.info("❌ Tests étape par étape: ÉCHEC")
    
    logger.info("📋 Vérifiez training_step_by_step.log pour plus de détails")
