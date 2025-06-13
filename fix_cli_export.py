#!/usr/bin/env python3
"""
Script pour corriger le problème CLI export sur RunPod
"""

import os
import sys
from pathlib import Path

def check_and_fix_export_module():
    """Vérifier et corriger le module export"""
    export_file = Path("retfound/cli/commands/export.py")
    
    if not export_file.exists():
        print(f"❌ Fichier {export_file} n'existe pas!")
        return False
    
    # Lire le contenu
    with open(export_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier si add_subparser existe
    if 'def add_subparser(' in content:
        print("✅ Fonction add_subparser trouvée dans export.py")
        
        # Vérifier s'il y a des erreurs de syntaxe
        lines = content.split('\n')
        in_add_subparser = False
        for i, line in enumerate(lines):
            if 'def add_subparser(' in line:
                in_add_subparser = True
                print(f"Ligne {i+1}: {line}")
            elif in_add_subparser and line.strip().startswith('def '):
                break
            elif in_add_subparser and line.strip():
                print(f"Ligne {i+1}: {line}")
        
        return True
    else:
        print("❌ Fonction add_subparser manquante dans export.py")
        
        # Ajouter la fonction
        add_subparser_code = '''

def add_subparser(subparsers):
    """Add export subcommand to parser"""
    parser = subparsers.add_parser(
        'export',
        help='Export RETFound model to various formats',
        description='Export trained RETFound model for deployment'
    )
    
    add_export_args(parser)
    parser.set_defaults(func=run_export)
    
    return parser
'''
        
        # Ajouter à la fin du fichier
        with open(export_file, 'a', encoding='utf-8') as f:
            f.write(add_subparser_code)
        
        print("✅ Fonction add_subparser ajoutée à export.py")
        return True

def test_import():
    """Tester l'import du module"""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from retfound.cli.commands import export
        
        if hasattr(export, 'add_subparser'):
            print("✅ Module export.add_subparser accessible")
            return True
        else:
            print("❌ Module export.add_subparser non accessible")
            return False
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def clear_python_cache():
    """Nettoyer le cache Python"""
    import shutil
    
    cache_dirs = []
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                cache_dirs.append(os.path.join(root, d))
    
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Cache supprimé: {cache_dir}")
        except Exception as e:
            print(f"⚠️ Erreur suppression cache {cache_dir}: {e}")

def main():
    print("🔧 Correction du problème CLI export...")
    
    # 1. Vérifier le fichier
    if not check_and_fix_export_module():
        return 1
    
    # 2. Nettoyer le cache
    clear_python_cache()
    
    # 3. Tester l'import
    if test_import():
        print("✅ Problème résolu!")
        return 0
    else:
        print("❌ Problème persiste")
        return 1

if __name__ == '__main__':
    sys.exit(main())
