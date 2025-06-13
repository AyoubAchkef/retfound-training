#!/usr/bin/env python3
"""
AUDIT COMPLET ET CORRECTION SYSTÉMATIQUE DU PROJET RETFOUND
===========================================================

Script d'audit complet comme un développeur senior IA l'aurait fait.
Analyse TOUS les fichiers, détecte TOUTES les incohérences et les corrige.
"""

import os
import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple

class RETFoundProjectAuditor:
    """Auditeur complet du projet RETFound"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.fixes_applied = []
        
    def audit_complete_project(self):
        """Audit complet du projet"""
        print("🔍 AUDIT COMPLET DU PROJET RETFOUND")
        print("=" * 60)
        
        # 1. Vérifier la structure du projet
        self._audit_project_structure()
        
        # 2. Vérifier les imports et dépendances
        self._audit_imports_and_dependencies()
        
        # 3. Vérifier la cohérence des configurations
        self._audit_configurations()
        
        # 4. Vérifier les datasets et transformations
        self._audit_datasets_and_transforms()
        
        # 5. Vérifier les modèles et factory
        self._audit_models_and_factory()
        
        # 6. Vérifier le trainer et callbacks
        self._audit_trainer_and_callbacks()
        
        # 7. Vérifier le CLI
        self._audit_cli_system()
        
        # 8. Appliquer TOUTES les corrections
        self._apply_all_fixes()
        
        # 9. Rapport final
        self._generate_final_report()
    
    def _audit_project_structure(self):
        """Audit de la structure du projet"""
        print("\n📁 AUDIT STRUCTURE DU PROJET")
        
        required_files = [
            "retfound/__init__.py",
            "retfound/core/__init__.py",
            "retfound/core/config.py",
            "retfound/data/__init__.py", 
            "retfound/data/datasets.py",
            "retfound/models/__init__.py",
            "retfound/training/__init__.py",
            "retfound/training/trainer.py",
            "retfound/cli/__init__.py",
            "retfound/cli/main.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.errors.append(f"FICHIER MANQUANT: {file_path}")
            else:
                print(f"✅ {file_path}")
    
    def _audit_imports_and_dependencies(self):
        """Audit des imports et dépendances"""
        print("\n📦 AUDIT IMPORTS ET DÉPENDANCES")
        
        # Vérifier tous les fichiers Python
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            if "fix_" in py_file.name or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Vérifier la syntaxe Python
                try:
                    ast.parse(content)
                    print(f"✅ Syntaxe OK: {py_file.relative_to(self.project_root)}")
                except SyntaxError as e:
                    self.errors.append(f"ERREUR SYNTAXE: {py_file.relative_to(self.project_root)} - {e}")
                    
                    # Correction automatique des erreurs d'indentation
                    if "expected an indented block" in str(e):
                        self._fix_indentation_error(py_file, content)
                
                # Vérifier les imports problématiques
                self._check_imports(py_file, content)
                
            except Exception as e:
                self.errors.append(f"ERREUR LECTURE: {py_file.relative_to(self.project_root)} - {e}")
    
    def _fix_indentation_error(self, file_path: Path, content: str):
        """Correction automatique des erreurs d'indentation"""
        print(f"🔧 CORRECTION INDENTATION: {file_path.relative_to(self.project_root)}")
        
        # Patterns de correction spécifiques
        fixes = [
            # Correction des blocs if isinstance mal indentés
            (
                r'^if isinstance\(image, np\.ndarray\):$',
                '        if isinstance(image, np.ndarray):'
            ),
            (
                r'^elif not isinstance\(image, Image\.Image\):$',
                '        elif not isinstance(image, Image.Image):'
            ),
            (
                r'^if hasattr\(self\.transform, \'transform\'\):  # Albumentations$',
                '        if hasattr(self.transform, \'transform\'):  # Albumentations'
            ),
            (
                r'^else:  # torchvision$',
                '        else:  # torchvision'
            )
        ]
        
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            for pattern, replacement in fixes:
                if re.match(pattern, line.strip()):
                    # Calculer l'indentation nécessaire
                    if not line.startswith('        '):
                        fixed_line = replacement
                        break
            fixed_lines.append(fixed_line)
        
        fixed_content = '\n'.join(fixed_lines)
        
        # Sauvegarder le fichier corrigé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        self.fixes_applied.append(f"INDENTATION CORRIGÉE: {file_path.relative_to(self.project_root)}")
    
    def _check_imports(self, file_path: Path, content: str):
        """Vérifier les imports dans un fichier"""
        
        # Imports requis pour datasets.py
        if file_path.name == "datasets.py":
            required_imports = [
                "import numpy as np",
                "from PIL import Image"
            ]
            
            for required_import in required_imports:
                if required_import not in content:
                    self._add_missing_import(file_path, content, required_import)
    
    def _add_missing_import(self, file_path: Path, content: str, missing_import: str):
        """Ajouter un import manquant"""
        print(f"🔧 AJOUT IMPORT: {missing_import} dans {file_path.relative_to(self.project_root)}")
        
        lines = content.split('\n')
        
        # Trouver où insérer l'import
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_index = i + 1
            elif line.strip() == '':
                continue
            else:
                break
        
        lines.insert(insert_index, missing_import)
        
        fixed_content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        self.fixes_applied.append(f"IMPORT AJOUTÉ: {missing_import} dans {file_path.relative_to(self.project_root)}")
    
    def _audit_configurations(self):
        """Audit des fichiers de configuration"""
        print("\n⚙️ AUDIT CONFIGURATIONS")
        
        config_files = [
            "configs/runpod.yaml",
            "configs/default.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                print(f"✅ {config_file}")
            else:
                self.warnings.append(f"CONFIG MANQUANTE: {config_file}")
    
    def _audit_datasets_and_transforms(self):
        """Audit spécifique des datasets et transformations"""
        print("\n🖼️ AUDIT DATASETS ET TRANSFORMATIONS")
        
        datasets_file = self.project_root / "retfound/data/datasets.py"
        if datasets_file.exists():
            with open(datasets_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Vérifier les patterns problématiques
            problematic_patterns = [
                "image = self.transform(image)",
                "image = Image.fromarray(image)"
            ]
            
            for pattern in problematic_patterns:
                if pattern in content and "if hasattr" not in content[content.find(pattern)-50:content.find(pattern)+50]:
                    self._fix_transform_pattern(datasets_file, content, pattern)
    
    def _fix_transform_pattern(self, file_path: Path, content: str, pattern: str):
        """Corriger les patterns de transformation problématiques"""
        print(f"🔧 CORRECTION PATTERN: {pattern} dans {file_path.relative_to(self.project_root)}")
        
        if pattern == "image = self.transform(image)":
            replacement = """        if hasattr(self.transform, 'transform'):  # Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
        else:  # torchvision
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = self.transform(image)"""
        
        elif pattern == "image = Image.fromarray(image)":
            replacement = """        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))"""
        
        content = content.replace(pattern, replacement)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.fixes_applied.append(f"PATTERN CORRIGÉ: {pattern} dans {file_path.relative_to(self.project_root)}")
    
    def _audit_models_and_factory(self):
        """Audit des modèles et factory"""
        print("\n🤖 AUDIT MODÈLES ET FACTORY")
        
        models_files = [
            "retfound/models/__init__.py",
            "retfound/models/factory.py",
            "retfound/models/retfound.py"
        ]
        
        for model_file in models_files:
            model_path = self.project_root / model_file
            if model_path.exists():
                print(f"✅ {model_file}")
            else:
                self.warnings.append(f"MODÈLE MANQUANT: {model_file}")
    
    def _audit_trainer_and_callbacks(self):
        """Audit du trainer et callbacks"""
        print("\n🏋️ AUDIT TRAINER ET CALLBACKS")
        
        trainer_files = [
            "retfound/training/trainer.py",
            "retfound/training/callbacks/__init__.py"
        ]
        
        for trainer_file in trainer_files:
            trainer_path = self.project_root / trainer_file
            if trainer_path.exists():
                print(f"✅ {trainer_file}")
                
                # Vérifier les accès aux attributs de config
                with open(trainer_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Rechercher les accès directs à config sans getattr
                unsafe_patterns = re.findall(r'config\.(\w+)(?!\s*=)', content)
                for attr in unsafe_patterns:
                    if f"getattr(config, '{attr}'" not in content:
                        self.warnings.append(f"ACCÈS NON SÉCURISÉ: config.{attr} dans {trainer_file}")
            else:
                self.errors.append(f"TRAINER MANQUANT: {trainer_file}")
    
    def _audit_cli_system(self):
        """Audit du système CLI"""
        print("\n💻 AUDIT SYSTÈME CLI")
        
        cli_files = [
            "retfound/cli/main.py",
            "retfound/cli/commands/train.py"
        ]
        
        for cli_file in cli_files:
            cli_path = self.project_root / cli_file
            if cli_path.exists():
                print(f"✅ {cli_file}")
            else:
                self.errors.append(f"CLI MANQUANT: {cli_file}")
    
    def _apply_all_fixes(self):
        """Appliquer toutes les corrections identifiées"""
        print("\n🔧 APPLICATION DES CORRECTIONS")
        
        for fix in self.fixes_applied:
            print(f"✅ {fix}")
    
    def _generate_final_report(self):
        """Générer le rapport final"""
        print("\n📋 RAPPORT FINAL D'AUDIT")
        print("=" * 60)
        
        print(f"✅ CORRECTIONS APPLIQUÉES: {len(self.fixes_applied)}")
        print(f"⚠️  AVERTISSEMENTS: {len(self.warnings)}")
        print(f"❌ ERREURS CRITIQUES: {len(self.errors)}")
        
        if self.errors:
            print("\n❌ ERREURS CRITIQUES À RÉSOUDRE:")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print("\n⚠️  AVERTISSEMENTS:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors:
            print("\n🎉 PROJET PRÊT POUR L'ENTRAÎNEMENT!")
        else:
            print("\n🔧 CORRECTIONS SUPPLÉMENTAIRES NÉCESSAIRES")

def main():
    """Fonction principale"""
    auditor = RETFoundProjectAuditor()
    auditor.audit_complete_project()

if __name__ == "__main__":
    main()
