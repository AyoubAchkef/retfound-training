"""
Script pour supprimer tous les warnings du CLI RETFound
======================================================

Ce script configure la suppression de tous les warnings pour un affichage propre.
"""

import warnings
import logging
import os
import sys

def suppress_all_warnings():
    """Supprime tous les warnings pour un affichage propre"""
    
    # 1. Supprimer tous les warnings Python
    warnings.filterwarnings("ignore")
    
    # 2. Supprimer les warnings spécifiques
    
    # Albumentations warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
    warnings.filterwarnings("ignore", message=".*alpha_affine.*")
    warnings.filterwarnings("ignore", message=".*fill_value.*")
    warnings.filterwarnings("ignore", message=".*A new version of Albumentations.*")
    
    # Pydantic warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
    
    # Scikit-learn warnings
    warnings.filterwarnings("ignore", message=".*scikit-learn not available.*")
    
    # TensorRT warnings
    warnings.filterwarnings("ignore", message=".*TensorRT dependencies not available.*")
    
    # 3. Configurer les loggers pour supprimer les warnings de configuration
    
    # Logger de configuration
    config_logger = logging.getLogger('retfound.core.config')
    config_logger.setLevel(logging.ERROR)  # Supprime INFO et WARNING
    
    # Logger CLI
    cli_logger = logging.getLogger('retfound.cli.main')
    cli_logger.setLevel(logging.ERROR)  # Supprime les warnings d'export
    
    # 4. Variables d'environnement pour supprimer les warnings
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Supprime les warnings de mise à jour
    os.environ['PYTHONWARNINGS'] = 'ignore'  # Supprime tous les warnings Python
    
    # 5. Supprimer les warnings de deprecation
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    print("✅ Tous les warnings ont été supprimés pour un affichage propre")

if __name__ == "__main__":
    suppress_all_warnings()
