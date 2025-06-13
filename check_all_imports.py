#!/usr/bin/env python3
"""
Check All Imports - RETFound
============================

Script pour vérifier tous les imports et détecter les problèmes potentiels
avant qu'ils ne causent des erreurs sur RunPod.
"""

import ast
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Set, Tuple

def find_python_files(directory: Path) -> List[Path]:
    """Trouve tous les fichiers Python dans le projet"""
    python_files = []
    for file_path in directory.rglob("*.py"):
        # Ignorer certains dossiers
        if any(part in str(file_path) for part in [
            "__pycache__", ".git", "node_modules", "dist", "build"
        ]):
            continue
        python_files.append(file_path)
    return python_files

def extract_imports(file_path: Path) -> Tuple[List[str], List[str]]:
    """Extrait les imports d'un fichier Python"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        from_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    from_imports.append(f"{module}.{alias.name}")
        
        return imports, from_imports
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse de {file_path}: {e}")
        return [], []

def check_typing_imports(file_path: Path) -> List[str]:
    """Vérifie les imports typing manquants"""
    problems = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Types couramment utilisés
        typing_types = [
            'Dict', 'List', 'Tuple', 'Optional', 'Union', 'Any', 'Callable',
            'Set', 'FrozenSet', 'Sequence', 'Mapping', 'Iterable', 'Iterator'
        ]
        
        # Vérifier si des types sont utilisés sans import
        for type_name in typing_types:
            if f"{type_name}[" in content or f"-> {type_name}" in content or f": {type_name}" in content:
                # Vérifier si le type est importé
                if f"from typing import" not in content or type_name not in content.split("from typing import")[1].split("\n")[0]:
                    if f"import typing" not in content:
                        problems.append(f"Type '{type_name}' utilisé mais pas importé depuis typing")
        
    except Exception as e:
        problems.append(f"Erreur lors de la vérification: {e}")
    
    return problems

def test_import_file(file_path: Path) -> List[str]:
    """Teste l'import d'un fichier Python"""
    problems = []
    
    try:
        # Convertir le chemin en module Python
        relative_path = file_path.relative_to(Path.cwd())
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        
        # Ignorer __init__.py dans le nom du module
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        
        module_name = ".".join(module_parts)
        
        # Ignorer certains modules
        if any(part in module_name for part in ["test_", "demo", "benchmark"]):
            return problems
        
        # Tenter d'importer le module
        __import__(module_name)
    except ValueError as e:
        # Problème de chemin - ignorer
        return problems
    except ImportError as e:
        problems.append(f"ImportError: {e}")
    except NameError as e:
        problems.append(f"NameError: {e}")
    except SyntaxError as e:
        problems.append(f"SyntaxError: {e}")
    except Exception as e:
        problems.append(f"Autre erreur: {e}")
    
    return problems

def check_specific_patterns(file_path: Path) -> List[str]:
    """Vérifie des patterns spécifiques problématiques"""
    problems = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Vérifier les imports relatifs problématiques
            if "from ." in line and "__init__" not in str(file_path):
                if "from ..core" in line or "from ...core" in line:
                    problems.append(f"Ligne {i}: Import relatif potentiellement problématique: {line.strip()}")
            
            # Vérifier les types non importés
            if ": Dict[" in line or "-> Dict[" in line:
                if "from typing import" not in content or "Dict" not in content:
                    problems.append(f"Ligne {i}: 'Dict' utilisé sans import")
            
            if ": List[" in line or "-> List[" in line:
                if "from typing import" not in content or "List" not in content:
                    problems.append(f"Ligne {i}: 'List' utilisé sans import")
            
            # Vérifier les imports de callbacks manquants
            if "Callback" in line and "import" not in line:
                if "from retfound.training.callbacks import" not in content:
                    problems.append(f"Ligne {i}: Callback utilisé sans import approprié")
    
    except Exception as e:
        problems.append(f"Erreur lors de la vérification des patterns: {e}")
    
    return problems

def main():
    """Fonction principale"""
    print("🔍 Vérification de tous les imports RETFound")
    print("=" * 50)
    
    # Trouver tous les fichiers Python
    python_files = find_python_files(Path("retfound"))
    print(f"📁 Fichiers Python trouvés: {len(python_files)}")
    
    all_problems = {}
    total_problems = 0
    
    for file_path in python_files:
        print(f"\n📄 Vérification: {file_path}")
        
        file_problems = []
        
        # 1. Vérifier les imports typing
        typing_problems = check_typing_imports(file_path)
        file_problems.extend(typing_problems)
        
        # 2. Vérifier les patterns spécifiques
        pattern_problems = check_specific_patterns(file_path)
        file_problems.extend(pattern_problems)
        
        # 3. Tester l'import du fichier
        import_problems = test_import_file(file_path)
        file_problems.extend(import_problems)
        
        if file_problems:
            all_problems[str(file_path)] = file_problems
            total_problems += len(file_problems)
            print(f"  ❌ {len(file_problems)} problème(s) détecté(s)")
            for problem in file_problems:
                print(f"    • {problem}")
        else:
            print(f"  ✅ Aucun problème détecté")
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DE LA VÉRIFICATION")
    print("=" * 50)
    
    if total_problems == 0:
        print("🎉 AUCUN PROBLÈME DÉTECTÉ !")
        print("✅ Tous les imports semblent corrects")
        return True
    else:
        print(f"⚠️  {total_problems} problème(s) détecté(s) dans {len(all_problems)} fichier(s)")
        
        print("\n🔧 PROBLÈMES À CORRIGER :")
        for file_path, problems in all_problems.items():
            print(f"\n📄 {file_path}:")
            for problem in problems:
                print(f"  • {problem}")
        
        print("\n💡 SUGGESTIONS DE CORRECTION :")
        print("1. Ajouter les imports typing manquants")
        print("2. Corriger les imports relatifs problématiques")
        print("3. Vérifier les callbacks manquants")
        print("4. Tester chaque correction individuellement")
        
        return False

def check_critical_files():
    """Vérifie spécifiquement les fichiers critiques"""
    print("\n🎯 VÉRIFICATION DES FICHIERS CRITIQUES")
    print("-" * 40)
    
    critical_files = [
        "retfound/cli/main.py",
        "retfound/cli/commands/train.py",
        "retfound/training/callbacks/__init__.py",
        "retfound/utils/reproducibility.py",
        "retfound/monitoring/demo.py",
        "retfound/monitoring/server.py"
    ]
    
    for file_path in critical_files:
        path = Path(file_path)
        if path.exists():
            print(f"📄 {file_path}")
            try:
                # Test d'import direct
                if file_path == "retfound/cli/main.py":
                    from retfound.cli.main import create_parser
                    print("  ✅ CLI import OK")
                elif file_path == "retfound/training/callbacks/__init__.py":
                    from retfound.training.callbacks import VisualizationCallback, WandbCallback
                    print("  ✅ Callbacks import OK")
                elif file_path == "retfound/utils/reproducibility.py":
                    from retfound.utils.reproducibility import log_random_states
                    print("  ✅ Reproducibility import OK")
                elif file_path == "retfound/monitoring/demo.py":
                    # Test sans exécution
                    print("  ✅ Demo file exists")
                else:
                    print("  ✅ File exists")
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
        else:
            print(f"  ❌ Fichier manquant: {file_path}")

if __name__ == "__main__":
    try:
        # Vérification générale
        success = main()
        
        # Vérification des fichiers critiques
        check_critical_files()
        
        print(f"\n{'='*50}")
        if success:
            print("🎉 VÉRIFICATION TERMINÉE - AUCUN PROBLÈME !")
            sys.exit(0)
        else:
            print("⚠️  VÉRIFICATION TERMINÉE - PROBLÈMES DÉTECTÉS")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Vérification interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1)
