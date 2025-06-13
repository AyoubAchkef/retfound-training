#!/usr/bin/env python3
"""
Script de Validation Complète - RETFound Training Setup
======================================================

Ce script vérifie la cohérence complète du projet RETFound pour RunPod,
incluant la validation du frontend, backend, configurations et intégration.
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util

# Colors for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.PURPLE}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.PURPLE}{Colors.BOLD}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.PURPLE}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        
    def add_success(self, message: str):
        self.passed += 1
        print_success(message)
        
    def add_error(self, message: str):
        self.failed += 1
        self.errors.append(message)
        print_error(message)
        
    def add_warning(self, message: str):
        self.warnings += 1
        print_warning(message)
        
    def summary(self):
        print_header("RÉSUMÉ DE VALIDATION")
        print(f"{Colors.GREEN}✓ Tests réussis: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}✗ Tests échoués: {self.failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠ Avertissements: {self.warnings}{Colors.RESET}")
        
        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}Erreurs détectées:{Colors.RESET}")
            for error in self.errors:
                print(f"  - {error}")
        
        return self.failed == 0

def validate_project_structure(result: ValidationResult):
    """Valide la structure du projet"""
    print_header("VALIDATION DE LA STRUCTURE DU PROJET")
    
    required_files = [
        "configs/dataset_v6.1.yaml",
        "configs/runpod.yaml",
        "configs/default.yaml",
        "retfound/__init__.py",
        "retfound/core/constants.py",
        "retfound/monitoring/server.py",
        "retfound/monitoring/api_routes.py",
        "retfound/monitoring/data_manager.py",
        "retfound/monitoring/frontend/package.json",
        "retfound/monitoring/frontend/src/App.tsx",
        "retfound/monitoring/frontend/src/hooks/useWebSocket.ts",
        "retfound/monitoring/frontend/src/store/monitoring.ts",
        "requirements.txt",
        "requirements-runpod.txt",
        ".env.runpod",
        "scripts/setup_runpod_complete.sh",
        "RUNPOD_TRAINING_GUIDE.md",
        "FRONTEND_BACKEND_INTEGRATION.md"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            result.add_success(f"Fichier trouvé: {file_path}")
        else:
            result.add_error(f"Fichier manquant: {file_path}")
    
    # Vérifier les répertoires
    required_dirs = [
        "retfound/monitoring/frontend/src/components",
        "retfound/monitoring/frontend/src/hooks",
        "retfound/monitoring/frontend/src/store",
        "retfound/core",
        "retfound/data",
        "retfound/models",
        "retfound/training"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            result.add_success(f"Répertoire trouvé: {dir_path}")
        else:
            result.add_error(f"Répertoire manquant: {dir_path}")

def validate_configurations(result: ValidationResult):
    """Valide les fichiers de configuration"""
    print_header("VALIDATION DES CONFIGURATIONS")
    
    # Valider dataset_v6.1.yaml
    try:
        with open("configs/dataset_v6.1.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Vérifier les champs critiques
        if config.get('dataset', {}).get('unified_classes'):
            result.add_success("Configuration unified_classes activée")
        else:
            result.add_error("Configuration unified_classes manquante")
            
        if config.get('model', {}).get('num_classes') == 28:
            result.add_success("Nombre de classes correct: 28")
        else:
            result.add_error(f"Nombre de classes incorrect: {config.get('model', {}).get('num_classes')}")
            
    except Exception as e:
        result.add_error(f"Erreur lors de la lecture de dataset_v6.1.yaml: {e}")
    
    # Valider runpod.yaml
    try:
        with open("configs/runpod.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        if "/workspace/datasets/DATASET_CLASSIFICATION" in str(config.get('dataset_path', '')):
            result.add_success("Chemin dataset RunPod correct")
        else:
            result.add_error("Chemin dataset RunPod incorrect")
            
        if config.get('optimizations', {}).get('amp_dtype') == 'bfloat16':
            result.add_success("Configuration A100 bfloat16 activée")
        else:
            result.add_warning("Configuration bfloat16 non trouvée")
            
    except Exception as e:
        result.add_error(f"Erreur lors de la lecture de runpod.yaml: {e}")

def validate_frontend_config(result: ValidationResult):
    """Valide la configuration frontend"""
    print_header("VALIDATION DU FRONTEND")
    
    # Vérifier package.json
    try:
        with open("retfound/monitoring/frontend/package.json", 'r') as f:
            package = json.load(f)
        
        required_deps = [
            "react", "react-dom", "typescript", "vite", 
            "recharts", "zustand", "framer-motion", "lucide-react"
        ]
        
        all_deps = {**package.get('dependencies', {}), **package.get('devDependencies', {})}
        
        for dep in required_deps:
            if dep in all_deps:
                result.add_success(f"Dépendance frontend trouvée: {dep}")
            else:
                result.add_error(f"Dépendance frontend manquante: {dep}")
                
    except Exception as e:
        result.add_error(f"Erreur lors de la lecture de package.json: {e}")
    
    # Vérifier les fichiers de configuration Vite
    vite_configs = [
        "retfound/monitoring/frontend/vite.config.ts",
        "retfound/monitoring/frontend/vite.config.runpod.ts"
    ]
    
    for config_file in vite_configs:
        if Path(config_file).exists():
            result.add_success(f"Configuration Vite trouvée: {config_file}")
        else:
            result.add_error(f"Configuration Vite manquante: {config_file}")
    
    # Vérifier .env.runpod
    if Path("retfound/monitoring/frontend/.env.runpod").exists():
        result.add_success("Configuration environnement frontend trouvée")
        
        try:
            with open("retfound/monitoring/frontend/.env.runpod", 'r') as f:
                content = f.read()
                if "VITE_API_URL=http://0.0.0.0:8000" in content:
                    result.add_success("URL API frontend correcte")
                else:
                    result.add_error("URL API frontend incorrecte")
        except Exception as e:
            result.add_error(f"Erreur lecture .env.runpod frontend: {e}")
    else:
        result.add_error("Fichier .env.runpod frontend manquant")

def validate_backend_integration(result: ValidationResult):
    """Valide l'intégration backend"""
    print_header("VALIDATION DU BACKEND")
    
    # Vérifier les imports Python
    try:
        sys.path.append('.')
        
        # Test d'import des modules principaux
        modules_to_test = [
            "retfound.core.constants",
            "retfound.monitoring.server",
            "retfound.monitoring.api_routes",
            "retfound.monitoring.data_manager"
        ]
        
        for module_name in modules_to_test:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    result.add_success(f"Module Python trouvé: {module_name}")
                else:
                    result.add_error(f"Module Python manquant: {module_name}")
            except Exception as e:
                result.add_error(f"Erreur import {module_name}: {e}")
                
    except Exception as e:
        result.add_error(f"Erreur lors de la validation des imports: {e}")
    
    # Vérifier les constantes critiques
    try:
        from retfound.core.constants import (
            NUM_TOTAL_CLASSES, UNIFIED_CLASS_NAMES, 
            CRITICAL_CONDITIONS, DATASET_V61_STATS
        )
        
        if NUM_TOTAL_CLASSES == 28:
            result.add_success("Nombre total de classes correct: 28")
        else:
            result.add_error(f"Nombre total de classes incorrect: {NUM_TOTAL_CLASSES}")
            
        if len(UNIFIED_CLASS_NAMES) == 28:
            result.add_success("Liste des classes unifiées correcte")
        else:
            result.add_error(f"Liste des classes incorrecte: {len(UNIFIED_CLASS_NAMES)}")
            
        if len(CRITICAL_CONDITIONS) >= 7:
            result.add_success(f"Conditions critiques définies: {len(CRITICAL_CONDITIONS)}")
        else:
            result.add_warning(f"Peu de conditions critiques: {len(CRITICAL_CONDITIONS)}")
            
    except Exception as e:
        result.add_error(f"Erreur lors de la validation des constantes: {e}")

def validate_monitoring_integration(result: ValidationResult):
    """Valide l'intégration du monitoring"""
    print_header("VALIDATION DU MONITORING")
    
    # Vérifier les composants frontend
    frontend_components = [
        "retfound/monitoring/frontend/src/components/Dashboard/Header.tsx",
        "retfound/monitoring/frontend/src/components/Dashboard/MetricsGrid.tsx",
        "retfound/monitoring/frontend/src/components/Charts/MetricsChart.tsx",
        "retfound/monitoring/frontend/src/components/Charts/ClassPerformance.tsx",
        "retfound/monitoring/frontend/src/components/Monitoring/CriticalAlerts.tsx",
        "retfound/monitoring/frontend/src/components/Monitoring/GPUStats.tsx"
    ]
    
    for component in frontend_components:
        if Path(component).exists():
            result.add_success(f"Composant frontend trouvé: {Path(component).name}")
        else:
            result.add_error(f"Composant frontend manquant: {component}")
    
    # Vérifier le hook WebSocket
    websocket_file = "retfound/monitoring/frontend/src/hooks/useWebSocket.ts"
    if Path(websocket_file).exists():
        try:
            with open(websocket_file, 'r') as f:
                content = f.read()
                
            if "useWebSocket" in content:
                result.add_success("Hook WebSocket défini")
            else:
                result.add_error("Hook WebSocket non trouvé")
                
            if "reconnect" in content.lower():
                result.add_success("Logique de reconnexion WebSocket présente")
            else:
                result.add_warning("Logique de reconnexion WebSocket manquante")
                
        except Exception as e:
            result.add_error(f"Erreur lecture useWebSocket.ts: {e}")
    else:
        result.add_error("Fichier useWebSocket.ts manquant")

def validate_scripts_and_deployment(result: ValidationResult):
    """Valide les scripts de déploiement"""
    print_header("VALIDATION DES SCRIPTS DE DÉPLOIEMENT")
    
    # Vérifier les scripts
    scripts = [
        "scripts/setup_runpod_complete.sh",
        "scripts/setup_runpod.sh"
    ]
    
    for script in scripts:
        if Path(script).exists():
            result.add_success(f"Script trouvé: {script}")
            
            # Vérifier les permissions
            if os.access(script, os.X_OK):
                result.add_success(f"Script exécutable: {script}")
            else:
                result.add_warning(f"Script non exécutable: {script}")
        else:
            result.add_error(f"Script manquant: {script}")
    
    # Vérifier les fichiers d'environnement
    env_files = [
        ".env.runpod",
        "retfound/monitoring/frontend/.env.runpod"
    ]
    
    for env_file in env_files:
        if Path(env_file).exists():
            result.add_success(f"Fichier environnement trouvé: {env_file}")
        else:
            result.add_error(f"Fichier environnement manquant: {env_file}")

def validate_documentation(result: ValidationResult):
    """Valide la documentation"""
    print_header("VALIDATION DE LA DOCUMENTATION")
    
    docs = [
        "RUNPOD_TRAINING_GUIDE.md",
        "FRONTEND_BACKEND_INTEGRATION.md",
        "RUNPOD_INSTALLATION.md"
    ]
    
    for doc in docs:
        if Path(doc).exists():
            result.add_success(f"Documentation trouvée: {doc}")
            
            # Vérifier la taille (doit être substantielle)
            size = Path(doc).stat().st_size
            if size > 1000:  # Au moins 1KB
                result.add_success(f"Documentation substantielle: {doc} ({size} bytes)")
            else:
                result.add_warning(f"Documentation courte: {doc} ({size} bytes)")
        else:
            result.add_error(f"Documentation manquante: {doc}")

def validate_class_consistency(result: ValidationResult):
    """Valide la cohérence des classes à travers le projet"""
    print_header("VALIDATION DE LA COHÉRENCE DES CLASSES")
    
    try:
        # Vérifier les constantes
        from retfound.core.constants import (
            UNIFIED_CLASS_NAMES, NUM_FUNDUS_CLASSES, 
            NUM_OCT_CLASSES, NUM_TOTAL_CLASSES
        )
        
        # Vérifications arithmétiques
        if NUM_FUNDUS_CLASSES + NUM_OCT_CLASSES == NUM_TOTAL_CLASSES:
            result.add_success("Cohérence arithmétique des classes")
        else:
            result.add_error("Incohérence arithmétique des classes")
            
        if len(UNIFIED_CLASS_NAMES) == NUM_TOTAL_CLASSES:
            result.add_success("Cohérence liste des classes unifiées")
        else:
            result.add_error("Incohérence liste des classes unifiées")
            
        # Vérifier les préfixes
        fundus_count = sum(1 for name in UNIFIED_CLASS_NAMES if name.startswith('Fundus_'))
        oct_count = sum(1 for name in UNIFIED_CLASS_NAMES if name.startswith('OCT_'))
        
        if fundus_count == NUM_FUNDUS_CLASSES:
            result.add_success(f"Nombre correct de classes Fundus: {fundus_count}")
        else:
            result.add_error(f"Nombre incorrect de classes Fundus: {fundus_count}")
            
        if oct_count == NUM_OCT_CLASSES:
            result.add_success(f"Nombre correct de classes OCT: {oct_count}")
        else:
            result.add_error(f"Nombre incorrect de classes OCT: {oct_count}")
            
    except Exception as e:
        result.add_error(f"Erreur validation cohérence classes: {e}")

def validate_critical_conditions(result: ValidationResult):
    """Valide la configuration des conditions critiques"""
    print_header("VALIDATION DES CONDITIONS CRITIQUES")
    
    try:
        from retfound.core.constants import CRITICAL_CONDITIONS
        
        expected_conditions = [
            'RAO', 'RVO', 'Retinal_Detachment', 'CNV', 
            'DR_Proliferative', 'DME', 'Glaucoma_Positive'
        ]
        
        for condition in expected_conditions:
            if condition in CRITICAL_CONDITIONS:
                info = CRITICAL_CONDITIONS[condition]
                
                result.add_success(f"Condition critique définie: {condition}")
                
                if 'min_sensitivity' in info:
                    sensitivity = info['min_sensitivity']
                    if 0.9 <= sensitivity <= 1.0:
                        result.add_success(f"Seuil de sensibilité valide pour {condition}: {sensitivity}")
                    else:
                        result.add_warning(f"Seuil de sensibilité questionnable pour {condition}: {sensitivity}")
                else:
                    result.add_error(f"Seuil de sensibilité manquant pour {condition}")
                    
                if 'reason' in info:
                    result.add_success(f"Raison définie pour {condition}")
                else:
                    result.add_warning(f"Raison manquante pour {condition}")
            else:
                result.add_error(f"Condition critique manquante: {condition}")
                
    except Exception as e:
        result.add_error(f"Erreur validation conditions critiques: {e}")

def main():
    """Fonction principale de validation"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              VALIDATION COMPLÈTE RETFOUND v6.1               ║")
    print("║                     Setup RunPod + Frontend                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    result = ValidationResult()
    
    # Exécuter toutes les validations
    validate_project_structure(result)
    validate_configurations(result)
    validate_frontend_config(result)
    validate_backend_integration(result)
    validate_monitoring_integration(result)
    validate_scripts_and_deployment(result)
    validate_documentation(result)
    validate_class_consistency(result)
    validate_critical_conditions(result)
    
    # Afficher le résumé
    success = result.summary()
    
    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 VALIDATION RÉUSSIE ! Le projet est prêt pour RunPod.{Colors.RESET}")
        print(f"{Colors.GREEN}Vous pouvez maintenant lancer l'installation sur RunPod.{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ VALIDATION ÉCHOUÉE ! Corrigez les erreurs avant de continuer.{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
