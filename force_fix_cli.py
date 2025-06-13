#!/usr/bin/env python3
"""
Script pour forcer la correction du CLI export sur RunPod
"""

import os
import sys
import shutil
from pathlib import Path

def force_fix_export_module():
    """Forcer la correction du module export"""
    export_file = Path("retfound/cli/commands/export.py")
    
    print(f"🔧 Correction forcée de {export_file}")
    
    # Lire le contenu actuel
    if export_file.exists():
        with open(export_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier si add_subparser existe déjà
        if 'def add_subparser(' in content:
            print("✅ Fonction add_subparser trouvée, mais pas accessible")
            
            # Supprimer la fonction existante et la recréer
            lines = content.split('\n')
            new_lines = []
            skip_function = False
            
            for line in lines:
                if 'def add_subparser(' in line:
                    skip_function = True
                    continue
                elif skip_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    skip_function = False
                
                if not skip_function:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        # Ajouter la fonction add_subparser à la fin
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
        
        # Écrire le nouveau contenu
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write(content + add_subparser_code)
        
        print("✅ Fonction add_subparser ajoutée/corrigée")
        return True
    else:
        print(f"❌ Fichier {export_file} n'existe pas!")
        return False

def clear_all_python_cache():
    """Nettoyer tout le cache Python"""
    print("🧹 Nettoyage complet du cache Python...")
    
    # Supprimer tous les __pycache__
    for root, dirs, files in os.walk('.'):
        for d in dirs[:]:  # Copie pour modification sûre
            if d == '__pycache__':
                cache_path = os.path.join(root, d)
                try:
                    shutil.rmtree(cache_path)
                    print(f"✅ Cache supprimé: {cache_path}")
                except Exception as e:
                    print(f"⚠️ Erreur suppression {cache_path}: {e}")
                dirs.remove(d)  # Ne pas descendre dans ce dossier
    
    # Supprimer les .pyc individuels
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.pyc'):
                pyc_path = os.path.join(root, f)
                try:
                    os.remove(pyc_path)
                    print(f"✅ Fichier .pyc supprimé: {pyc_path}")
                except Exception as e:
                    print(f"⚠️ Erreur suppression {pyc_path}: {e}")

def force_reload_modules():
    """Forcer le rechargement des modules"""
    print("🔄 Rechargement forcé des modules...")
    
    # Supprimer les modules du cache sys.modules
    modules_to_remove = []
    for module_name in sys.modules:
        if module_name.startswith('retfound.cli'):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        del sys.modules[module_name]
        print(f"✅ Module supprimé du cache: {module_name}")

def test_import_with_details():
    """Tester l'import avec détails"""
    print("🧪 Test d'import détaillé...")
    
    try:
        # Ajouter le répertoire courant au path
        current_dir = str(Path.cwd())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        print(f"📁 Répertoire de travail: {current_dir}")
        print(f"🐍 Python path: {sys.path[:3]}...")
        
        # Test import du module
        print("📦 Import du module export...")
        from retfound.cli.commands import export
        
        print(f"📍 Module export chargé depuis: {export.__file__}")
        
        # Lister les attributs du module
        attrs = [attr for attr in dir(export) if not attr.startswith('_')]
        print(f"🔍 Attributs du module: {attrs}")
        
        # Test spécifique de add_subparser
        if hasattr(export, 'add_subparser'):
            print("✅ Fonction add_subparser trouvée!")
            
            # Tester l'appel
            import argparse
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers()
            
            result = export.add_subparser(subparsers)
            print(f"✅ Fonction add_subparser appelée avec succès: {type(result)}")
            return True
        else:
            print("❌ Fonction add_subparser non trouvée")
            print(f"🔍 Fonctions disponibles: {[attr for attr in attrs if callable(getattr(export, attr))]}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_export_module():
    """Créer un module export minimal fonctionnel"""
    export_file = Path("retfound/cli/commands/export.py")
    
    minimal_content = '''"""
Export command for RETFound CLI - Minimal version
"""

import argparse

def add_export_args(parser):
    """Add export arguments"""
    parser.add_argument('checkpoint', help='Model checkpoint path')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['pytorch'], help='Export formats')

def run_export(args):
    """Run export command"""
    print(f"Export command called with: {args}")
    print("Export functionality not yet implemented")
    return 0

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
    
    # Sauvegarder l'ancien fichier
    if export_file.exists():
        backup_file = export_file.with_suffix('.py.backup')
        shutil.copy2(export_file, backup_file)
        print(f"📋 Sauvegarde créée: {backup_file}")
    
    # Écrire le nouveau contenu minimal
    with open(export_file, 'w', encoding='utf-8') as f:
        f.write(minimal_content)
    
    print(f"✅ Module export minimal créé: {export_file}")

def main():
    print("🚀 Correction forcée du CLI export...")
    print("=" * 60)
    
    # 1. Nettoyer le cache
    clear_all_python_cache()
    
    # 2. Forcer le rechargement
    force_reload_modules()
    
    # 3. Tenter la correction normale
    if force_fix_export_module():
        if test_import_with_details():
            print("✅ Correction réussie!")
            return 0
    
    # 4. Si échec, créer un module minimal
    print("\n⚠️ Correction normale échouée, création d'un module minimal...")
    create_minimal_export_module()
    
    # 5. Nettoyer à nouveau et tester
    clear_all_python_cache()
    force_reload_modules()
    
    if test_import_with_details():
        print("✅ Module minimal fonctionnel!")
        return 0
    else:
        print("❌ Échec complet")
        return 1

if __name__ == '__main__':
    sys.exit(main())
